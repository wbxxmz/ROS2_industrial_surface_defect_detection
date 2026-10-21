"""阶段 6 §4.1：缺陷严重程度分级（纯函数、可复现、无跨帧状态）。

- 按 §4.1.2 的固定公式计算 ``score``，再映射为 ``0/1/2`` 三级；
- 所有阈值经 ``cfg``（即 ``config["severity"]``）传入，缺省取 §4.1.3 默认值；
- 返回**新的** ``Detection`` 列表（``dataclasses.replace``），不修改入参；
- 配置非法抛 ``ValueError``；单条检测数据非法时该条 ``severity=None`` 并记 warning。
"""

import logging
import math
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Union

from core.types import Detection
from detectors.opencv_detector import CLASS_MAP  # 阶段 3 §4.1 唯一权威映射表

logger = logging.getLogger("defect.severity")

# §4.1.1 取值集合
SEVERITY_MILD = 0
SEVERITY_MODERATE = 1
SEVERITY_SEVERE = 2
# 仅供报告/日志；画面统一用 ASCII 的 S0/S1/S2（Hershey 字体不支持中文）
SEVERITY_NAMES = {SEVERITY_MILD: "轻度", SEVERITY_MODERATE: "中度", SEVERITY_SEVERE: "重度"}

# §4.1.3 字段名
PARAM_NAMES = (
    "enabled",
    "weights",
    "area_ratio_ref",
    "conf_floor",
    "conf_ceiling",
    "class_bias",
    "grade_thresholds",
)
WEIGHT_NAMES = ("area", "confidence")
THRESHOLD_NAMES = ("mild_max", "severe_min")
DEFAULT_KEY = "default"

# §4.1.3 默认值
DEFAULT_WEIGHTS = {"area": 0.6, "confidence": 0.25}
DEFAULT_AREA_RATIO_REF = {0: 0.02, 1: 0.01, 2: 0.03, 3: 0.01, DEFAULT_KEY: 0.02}
DEFAULT_CLASS_BIAS = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, DEFAULT_KEY: 0.0}
DEFAULT_THRESHOLDS = {"mild_max": 0.35, "severe_min": 0.70}


def _require_bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} 必须是 bool，收到 {value!r}")
    return value


def _require_number(name: str, value: object) -> float:
    """有限数值（拒绝 bool / NaN / inf）。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是数值，收到 {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} 必须是有限数值，收到 {value!r}")
    return number


def _require_ratio(name: str, value: object) -> float:
    number = _require_number(name, value)
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{name} 必须落在 [0, 1]，收到 {number}")
    return number


def _require_open_unit(name: str, value: object) -> float:
    """(0, 1) 开区间。"""
    number = _require_number(name, value)
    if number <= 0.0 or number >= 1.0:
        raise ValueError(f"{name} 必须落在 (0, 1)，收到 {number}")
    return number


def _require_positive(name: str, value: object) -> float:
    number = _require_number(name, value)
    if number <= 0.0:
        raise ValueError(f"{name} 必须 > 0，收到 {number}")
    return number


def _resolve_class_key(name: str, key: object) -> Union[int, str]:
    """类字典的键：0~3 的 int，或 ``"default"``。"""
    if isinstance(key, bool):
        raise ValueError(f"{name} 的键不能是 bool，收到 {key!r}")
    if isinstance(key, str):
        if key == DEFAULT_KEY:
            return DEFAULT_KEY
        raise ValueError(f"{name} 只允许 class_id 0~3 或 'default'，收到 {key!r}")
    if isinstance(key, int):
        if key in CLASS_MAP:
            return key
        raise ValueError(f"{name} 的 class_id 越界：{key}（允许 {sorted(CLASS_MAP)}）")
    raise ValueError(f"{name} 的键必须是 int 或 'default'，收到 {type(key).__name__}")


def _parse_section(name: str, raw: object, allowed: tuple) -> dict:
    if not isinstance(raw, dict):
        raise ValueError(f"{name} 必须是 dict，收到 {type(raw).__name__}")
    unknown = sorted(set(raw) - set(allowed))
    if unknown:
        raise ValueError(f"{name} 含未知字段 {unknown}；允许字段 {allowed}")
    return raw


def _parse_class_map(name: str, raw: object, defaults: dict, check) -> dict:
    """解析按类的数值字典；``raw`` 为 None 时用默认值，显式给出时 ``default`` 必填。"""
    if raw is None:
        return dict(defaults)
    if not isinstance(raw, dict):
        raise ValueError(f"{name} 必须是 dict，收到 {type(raw).__name__}")
    result = {}
    for key, value in raw.items():
        result[_resolve_class_key(name, key)] = check(f"{name}[{key}]", value)
    if DEFAULT_KEY not in result:
        raise ValueError(f"{name} 缺少必填键 'default'（未配置类别的回退值）")
    return result


@dataclass(frozen=True)
class SeverityConfig:
    """§4.1.3 分级参数表。"""

    enabled: bool
    w_area: float
    w_conf: float
    area_ratio_ref: Dict[Union[int, str], float]
    conf_floor: float
    conf_ceiling: float
    class_bias: Dict[Union[int, str], float]
    mild_max: float
    severe_min: float

    @classmethod
    def from_dict(cls, cfg: Optional[dict]) -> "SeverityConfig":
        """校验并构造；``None``/``{}`` 视为**分级关闭**（§4.1.4）。"""
        if cfg is None or cfg == {}:
            return cls.disabled()
        if not isinstance(cfg, dict):
            raise ValueError(f"severity 配置必须是 dict，收到 {type(cfg).__name__}")
        unknown = sorted(set(cfg) - set(PARAM_NAMES))
        if unknown:
            raise ValueError(f"severity 配置含未知字段 {unknown}；允许字段见阶段 6 §4.1.3")

        enabled = _require_bool("enabled", cfg.get("enabled", True))

        weights = _parse_section("weights", cfg.get("weights", DEFAULT_WEIGHTS), WEIGHT_NAMES)
        w_area = _require_ratio("weights.area", weights.get("area", DEFAULT_WEIGHTS["area"]))
        w_conf = _require_ratio(
            "weights.confidence", weights.get("confidence", DEFAULT_WEIGHTS["confidence"])
        )
        if w_area + w_conf > 1.0 + 1e-9:
            raise ValueError(f"weights.area + weights.confidence 必须 <= 1，收到 {w_area + w_conf}")

        area_ratio_ref = _parse_class_map(
            "area_ratio_ref", cfg.get("area_ratio_ref"), DEFAULT_AREA_RATIO_REF, _require_positive
        )
        class_bias = _parse_class_map(
            "class_bias", cfg.get("class_bias"), DEFAULT_CLASS_BIAS, _require_ratio
        )

        conf_floor = _require_ratio("conf_floor", cfg.get("conf_floor", 0.25))
        conf_ceiling = _require_ratio("conf_ceiling", cfg.get("conf_ceiling", 0.95))
        if conf_floor >= conf_ceiling:
            raise ValueError(f"conf_floor 必须 < conf_ceiling，收到 {conf_floor} >= {conf_ceiling}")

        thresholds = _parse_section(
            "grade_thresholds", cfg.get("grade_thresholds", DEFAULT_THRESHOLDS), THRESHOLD_NAMES
        )
        mild_max = _require_open_unit(
            "grade_thresholds.mild_max", thresholds.get("mild_max", DEFAULT_THRESHOLDS["mild_max"])
        )
        severe_min = _require_open_unit(
            "grade_thresholds.severe_min",
            thresholds.get("severe_min", DEFAULT_THRESHOLDS["severe_min"]),
        )
        if mild_max >= severe_min:
            raise ValueError(f"mild_max 必须 < severe_min，收到 {mild_max} >= {severe_min}")

        return cls(
            enabled=enabled,
            w_area=w_area,
            w_conf=w_conf,
            area_ratio_ref=area_ratio_ref,
            conf_floor=conf_floor,
            conf_ceiling=conf_ceiling,
            class_bias=class_bias,
            mild_max=mild_max,
            severe_min=severe_min,
        )

    @classmethod
    def disabled(cls) -> "SeverityConfig":
        """分级关闭：结构合法但 ``enabled=False``。"""
        return cls(
            enabled=False,
            w_area=DEFAULT_WEIGHTS["area"],
            w_conf=DEFAULT_WEIGHTS["confidence"],
            area_ratio_ref=dict(DEFAULT_AREA_RATIO_REF),
            conf_floor=0.25,
            conf_ceiling=0.95,
            class_bias=dict(DEFAULT_CLASS_BIAS),
            mild_max=DEFAULT_THRESHOLDS["mild_max"],
            severe_min=DEFAULT_THRESHOLDS["severe_min"],
        )


def _score(detection: Detection, config: SeverityConfig) -> float:
    """§4.1.2 的加权分数；``clamp`` 到 [0, 1]。"""
    area_ref = config.area_ratio_ref.get(
        detection.class_id, config.area_ratio_ref[DEFAULT_KEY]
    )
    bias = config.class_bias.get(detection.class_id, config.class_bias[DEFAULT_KEY])
    s_area = min(1.0, max(0.0, float(detection.area_ratio) / area_ref))
    span = config.conf_ceiling - config.conf_floor
    s_conf = min(1.0, max(0.0, (float(detection.confidence) - config.conf_floor) / span))
    return min(1.0, max(0.0, config.w_area * s_area + config.w_conf * s_conf + bias))


def _grade_one(detection: Detection, config: SeverityConfig) -> Optional[int]:
    """单条分级；无法分级（非法数据）返回 None 并记 warning。"""
    if detection.class_id not in CLASS_MAP:
        logger.warning("无法分级：class_id=%r 不在阶段 3 §4.1 映射表内", detection.class_id)
        return None
    try:
        area_ratio = float(detection.area_ratio)
        confidence = float(detection.confidence)
    except (TypeError, ValueError):
        logger.warning("无法分级：area_ratio/confidence 非数值，class_id=%d", detection.class_id)
        return None
    if not (math.isfinite(area_ratio) and math.isfinite(confidence)):
        logger.warning(
            "无法分级：area_ratio/confidence 为 NaN/inf（%r, %r），class_id=%d",
            detection.area_ratio,
            detection.confidence,
            detection.class_id,
        )
        return None

    score = _score(detection, config)
    if score < config.mild_max:
        return SEVERITY_MILD
    if score < config.severe_min:
        return SEVERITY_MODERATE
    return SEVERITY_SEVERE


def grade(detections: List[Detection], cfg: Optional[dict] = None) -> List[Detection]:
    """为每条检测写入 ``severity``，返回新列表（入参不被修改）。

    :raises ValueError: ``detections`` 非 list/含非 ``Detection``，或 ``cfg`` 非法。
    """
    if detections is None or not isinstance(detections, list):
        raise ValueError(f"detections 必须是 list，收到 {type(detections).__name__}")
    for index, detection in enumerate(detections):
        if not isinstance(detection, Detection):
            raise ValueError(f"detections[{index}] 不是 Detection：{type(detection).__name__}")

    config = SeverityConfig.from_dict(cfg)
    if not config.enabled:  # §4.1.4：关闭时原样返回副本，severity 保持 None
        return [replace(detection) for detection in detections]
    return [
        replace(detection, severity=_grade_one(detection, config)) for detection in detections
    ]


__all__ = [
    "SEVERITY_MILD",
    "SEVERITY_MODERATE",
    "SEVERITY_SEVERE",
    "SEVERITY_NAMES",
    "SeverityConfig",
    "grade",
]
