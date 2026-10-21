"""阶段 5 §4.2 判定规则 + §4.3 统计汇总（无头、纯函数、无跨帧状态）。

- 规则参数全部经 ``cfg``（即 ``config["judge"]``）传入，缺省取 §4.2.2 默认值。
- ``class_id``/``class_name`` 复用阶段 3 §4.1 的 ``CLASS_MAP``，本模块只读、不重定义。
- 异常语义见 §5.1：非法入参与非法配置统一抛 ``ValueError``。
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from core.types import Detection
from core.verdict import Verdict
from detectors.opencv_detector import CLASS_MAP  # 阶段 3 §4.1 唯一权威映射表

logger = logging.getLogger("defect.judge")

# §4.2.2 顶层字段名（未知字段一律拒绝，避免拼写错误静默失效）
PARAM_NAMES = (
    "max_count",
    "max_area_ratio",
    "min_confidence",
    "min_area_px",
    "max_severity",
    "per_class",
)
# §4.2.2 per_class 子字段名
PER_CLASS_NAMES = ("max_count", "max_area_ratio")
# class_name → class_id（只由 CLASS_MAP 反推，不另建表）
NAME_TO_ID = {name: class_id for class_id, name in CLASS_MAP.items()}


# --------------------------------------------------------------------------- #
# 配置（§4.2.2）
# --------------------------------------------------------------------------- #
def _require_count(name: str, value: object) -> int:
    """非 bool 的 int 且 >= 0（bool 是 int 子类，须显式拒绝）。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} 必须是非 bool 的 int，收到 {value!r}")
    if value < 0:
        raise ValueError(f"{name} 必须 >= 0，收到 {value}")
    return value


def _require_ratio(name: str, value: object) -> float:
    """数值且落在 [0, 1]。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是数值，收到 {value!r}")
    if value < 0 or value > 1:
        raise ValueError(f"{name} 必须落在 [0, 1]，收到 {value}")
    return float(value)


def _resolve_class_id(key: object) -> int:
    """per_class 的键：0~3 的 int，或 CLASS_MAP 中的 class_name。"""
    if isinstance(key, bool):
        raise ValueError(f"per_class 的键不能是 bool，收到 {key!r}")
    if isinstance(key, int):
        if key in CLASS_MAP:
            return key
        raise ValueError(f"per_class 的 class_id 越界：{key}（允许 {sorted(CLASS_MAP)}）")
    if isinstance(key, str):
        if key in NAME_TO_ID:
            return NAME_TO_ID[key]
        raise ValueError(f"per_class 的 class_name 未知：{key!r}（允许 {sorted(NAME_TO_ID)}）")
    raise ValueError(f"per_class 的键必须是 int 或 str，收到 {type(key).__name__}")


def _require_severity_limit(name: str, value: object) -> int:
    """严重度上限：只接受 int 0/1/2（阶段 6 §4.1.5 扩展；null 由调用方处理）。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} 必须是 int（0/1/2）或 null，收到 {value!r}")
    if value not in (0, 1, 2):
        raise ValueError(f"{name} 只能是 0/1/2，收到 {value}")
    return value


@dataclass(frozen=True)
class PerClassRule:
    """类内约束；未配置的维度为 ``None``（不约束）。"""

    max_count: Optional[int] = None
    max_area_ratio: Optional[float] = None


@dataclass(frozen=True)
class JudgeConfig:
    """§4.2.2 判定参数表。"""

    max_count: int = 0
    max_area_ratio: float = 0.01
    min_confidence: float = 0.45
    min_area_px: int = 10
    max_severity: Optional[int] = None
    per_class: Dict[int, PerClassRule] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, cfg: Optional[dict]) -> "JudgeConfig":
        """校验并构造；``None`` 或 ``{}`` 取全部默认值。"""
        if cfg is None:
            cfg = {}
        if not isinstance(cfg, dict):
            raise ValueError(f"judge 配置必须是 dict，收到 {type(cfg).__name__}")
        unknown = sorted(set(cfg) - set(PARAM_NAMES))
        if unknown:
            raise ValueError(f"judge 配置含未知字段 {unknown}；允许字段见阶段 5 §4.2.2")

        raw_max_severity = cfg.get("max_severity")
        max_severity = (
            None
            if raw_max_severity is None  # 默认不启用严重度规则（阶段 5 兼容语义）
            else _require_severity_limit("max_severity", raw_max_severity)
        )

        raw_per_class = cfg.get("per_class")
        if raw_per_class is None:  # YAML 中只写键名不写值时的空值，等价于未配置
            raw_per_class = {}
        if not isinstance(raw_per_class, dict):
            raise ValueError(f"per_class 必须是 dict，收到 {type(raw_per_class).__name__}")

        per_class: Dict[int, PerClassRule] = {}
        for key, raw_rule in raw_per_class.items():
            class_id = _resolve_class_id(key)
            if class_id in per_class:  # 如同时写了 0 与 "scratch"
                raise ValueError(f"per_class 中 class_id={class_id} 被重复配置（{key!r}）")
            per_class[class_id] = _parse_per_class_rule(class_id, raw_rule)

        return cls(
            max_count=_require_count("max_count", cfg.get("max_count", 0)),
            max_area_ratio=_require_ratio("max_area_ratio", cfg.get("max_area_ratio", 0.01)),
            min_confidence=_require_ratio("min_confidence", cfg.get("min_confidence", 0.45)),
            min_area_px=_require_count("min_area_px", cfg.get("min_area_px", 10)),
            max_severity=max_severity,
            per_class=per_class,
        )


def _parse_per_class_rule(class_id: int, raw_rule: object) -> PerClassRule:
    """解析单个类内规则；未知子字段/非法值抛 ``ValueError``。"""
    if not isinstance(raw_rule, dict):
        raise ValueError(f"per_class[{class_id}] 必须是 dict，收到 {type(raw_rule).__name__}")
    unknown = sorted(set(raw_rule) - set(PER_CLASS_NAMES))
    if unknown:
        raise ValueError(f"per_class[{class_id}] 含未知字段 {unknown}；允许字段 {PER_CLASS_NAMES}")

    max_count = raw_rule.get("max_count")
    max_area_ratio = raw_rule.get("max_area_ratio")
    return PerClassRule(
        max_count=None if max_count is None else _require_count(f"per_class[{class_id}].max_count", max_count),
        max_area_ratio=(
            None
            if max_area_ratio is None
            else _require_ratio(f"per_class[{class_id}].max_area_ratio", max_area_ratio)
        ),
    )


# --------------------------------------------------------------------------- #
# 判定（§4.2.1）
# --------------------------------------------------------------------------- #
def judge(detections: List[Detection], cfg: Optional[dict] = None) -> Verdict:
    """按 §4.2.1 规则顺序求值，返回结构化判定结果。

    ``detections`` 必须为 list（``None`` 不视为"无结果"）；``cfg`` 为
    ``config["judge"]``，``None``/``{}`` 取默认值。非法入参抛 ``ValueError``。
    """
    if detections is None or not isinstance(detections, list):
        raise ValueError(f"detections 必须是 list，收到 {type(detections).__name__}")
    for index, detection in enumerate(detections):
        if not isinstance(detection, Detection):
            raise ValueError(f"detections[{index}] 不是 Detection：{type(detection).__name__}")

    config = JudgeConfig.from_dict(cfg)

    # §4.2.1 有效缺陷预筛；class_id 不在映射表内不计入（保持 total == sum(class_counts)）
    valid: List[Detection] = []
    for detection in detections:
        if detection.class_id not in CLASS_MAP:
            logger.warning("忽略 class_id=%d：不在阶段 3 §4.1 映射表内", detection.class_id)
            continue
        if detection.confidence < config.min_confidence:
            continue
        if detection.area_px < config.min_area_px:
            continue
        valid.append(detection)

    class_counts: Dict[int, int] = {class_id: 0 for class_id in CLASS_MAP}
    class_area_ratio: Dict[int, float] = {class_id: 0.0 for class_id in CLASS_MAP}
    for detection in valid:
        class_counts[detection.class_id] += 1
        class_area_ratio[detection.class_id] += float(detection.area_ratio)

    total_count = len(valid)
    total_area_px = sum(int(detection.area_px) for detection in valid)
    # 求和后 clip 到 [0,1]，但规则比较用 clip 后的值（§4.2.1）
    total_area_ratio = min(1.0, max(0.0, sum(float(d.area_ratio) for d in valid)))

    # §4.1.5 有效缺陷中的最高严重度；severity=None 的条目不参与（不当作 0）
    max_severity: Optional[int] = None
    for detection in valid:
        severity = detection.severity
        if severity is None:
            continue
        if isinstance(severity, bool) or not isinstance(severity, int) or severity not in (0, 1, 2):
            logger.warning("忽略非法 severity=%r：class_id=%d", severity, detection.class_id)
            continue
        max_severity = severity if max_severity is None else max(max_severity, severity)

    triggered: List[str] = []
    if total_count > config.max_count:
        triggered.append(f"count:{total_count}>{config.max_count}")
    if total_area_ratio > config.max_area_ratio:
        triggered.append(f"area:{total_area_ratio:.4f}>{config.max_area_ratio}")
    # 阶段 6 §4.1.5：severity:<实际最大值>>=<上限>；未配置上限或全部未分级时不触发
    if (
        config.max_severity is not None
        and max_severity is not None
        and max_severity > config.max_severity
    ):
        triggered.append(f"severity:{max_severity}>{config.max_severity}")

    # 类内规则只追加约束、不与全局规则互相收敛（§4.2.1 优先级语义）
    for class_id in sorted(config.per_class):
        rule = config.per_class[class_id]
        if rule.max_count is not None and class_counts[class_id] > rule.max_count:
            triggered.append(f"class:count:{class_id}:{class_counts[class_id]}>{rule.max_count}")
        ratio = min(1.0, max(0.0, class_area_ratio[class_id]))
        if rule.max_area_ratio is not None and ratio > rule.max_area_ratio:
            triggered.append(f"class:area:{class_id}:{ratio:.4f}>{rule.max_area_ratio}")

    is_pass = not triggered
    if is_pass:
        reason = "PASS：未检出有效缺陷" if total_count == 0 else "PASS：未触发判定规则"
    else:
        reason = "FAIL：" + "；".join(triggered)

    return Verdict(
        is_pass=is_pass,
        reason=reason,
        total_count=total_count,
        class_counts=class_counts,
        total_area_px=total_area_px,
        total_area_ratio=total_area_ratio,
        triggered=triggered,
        max_severity=max_severity,
    )


# --------------------------------------------------------------------------- #
# 批次汇总（§4.3）
# --------------------------------------------------------------------------- #
def summarize(verdicts: List[Verdict]) -> dict:
    """按 §4.3 汇总批次结果；空列表合法，``yield_rate`` 取 0.0（不除零）。"""
    if verdicts is None or not isinstance(verdicts, list):
        raise ValueError(f"verdicts 必须是 list，收到 {type(verdicts).__name__}")
    for index, verdict in enumerate(verdicts):
        if not isinstance(verdict, Verdict):
            raise ValueError(f"verdicts[{index}] 不是 Verdict：{type(verdict).__name__}")

    sample_count = len(verdicts)
    pass_count = sum(1 for verdict in verdicts if verdict.is_pass)
    fail_count = sample_count - pass_count

    class_counts: Dict[int, int] = {
        class_id: sum(verdict.class_counts.get(class_id, 0) for verdict in verdicts)
        for class_id in CLASS_MAP
    }
    return {
        "sample_count": sample_count,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "yield_rate": pass_count / sample_count if sample_count else 0.0,
        "total_count": sum(verdict.total_count for verdict in verdicts),
        "class_counts": class_counts,
        "total_area_px": sum(verdict.total_area_px for verdict in verdicts),
        "total_area_ratio": min(1.0, max(0.0, sum(v.total_area_ratio for v in verdicts))),
    }
