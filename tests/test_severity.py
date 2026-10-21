"""阶段 6 §7 验收：取值合法性、公式可复现、单调性、配置化与边界回退。"""

import math
import random
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import Detection  # noqa: E402
from detectors.opencv_detector import CLASS_MAP  # noqa: E402
from severity import SEVERITY_MILD, SEVERITY_MODERATE, SEVERITY_SEVERE, SeverityConfig, grade  # noqa: E402
from severity.grader import _score  # noqa: E402


def make_detection(
    class_id: int = 2,
    confidence: float = 0.666,
    area_ratio: float = 0.01815,
    area_px: int = 726,
    severity=None,
) -> Detection:
    """构造符合阶段 0 契约的 Detection（bbox 不参与分级）。"""
    return Detection(
        class_id=class_id,
        class_name=CLASS_MAP.get(class_id, "unknown"),
        bbox=(0, 0, 30, 25),
        area_px=area_px,
        area_ratio=area_ratio,
        confidence=confidence,
        mask=None,
        severity=severity,
    )


# --------------------------------------------------------------------------- #
# §7.1 取值合法性
# --------------------------------------------------------------------------- #
def test_range_random_legal_detections():
    rng = random.Random(20261021)
    detections = [
        make_detection(
            class_id=rng.choice([0, 1, 2, 3]),
            confidence=rng.uniform(0.0, 1.0),
            area_ratio=rng.uniform(0.0, 1.0),
        )
        for _ in range(50)
    ]
    result = grade(detections)
    assert len(result) == len(detections)
    assert all(item.severity in (None, 0, 1, 2) for item in result)


def test_empty_detections_returns_empty_list():
    assert grade([]) == []


# --------------------------------------------------------------------------- #
# §7.2 公式可复现且与文档示例一致
# --------------------------------------------------------------------------- #
def test_formula_matches_documented_example():
    detection = make_detection()  # §4.1.2：stain, area_ratio=0.01815, conf=0.666
    score = _score(detection, _config())
    assert score == pytest.approx(0.511, abs=1e-3)
    assert grade([detection], {"enabled": True})[0].severity == SEVERITY_MODERATE


def test_formula_is_deterministic():
    detections = [make_detection(class_id=class_id) for class_id in (0, 1, 2, 3)]
    first = grade(detections)
    second = grade(detections)
    assert [(d.class_id, d.severity) for d in first] == [(d.class_id, d.severity) for d in second]


def test_score_is_clamped_to_unit_interval():
    detection = make_detection(area_ratio=10.0, confidence=10.0)
    config = _config()
    # 默认权重和为 0.85，裸值上限即 0.85；权重和为 1 时饱和到 1.0
    assert _score(detection, config) == pytest.approx(0.85)
    saturated = SeverityConfig.from_dict(
        {"enabled": True, "weights": {"area": 0.75, "confidence": 0.25}}
    )
    assert _score(detection, saturated) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# §7.3 单调性
# --------------------------------------------------------------------------- #
def test_monotonic_in_area_ratio():
    severities = [
        grade([make_detection(area_ratio=ratio, confidence=0.666)], {"enabled": True})[0].severity
        for ratio in (0.0, 0.003, 0.01815, 0.03, 0.2)
    ]
    assert severities == sorted(severities)


def test_monotonic_in_confidence():
    severities = [
        grade([make_detection(area_ratio=0.01815, confidence=conf)], {"enabled": True})[0].severity
        for conf in (0.0, 0.3, 0.666, 0.95)
    ]
    assert severities == sorted(severities)


# --------------------------------------------------------------------------- #
# 配置化：阈值翻转与关闭语义
# --------------------------------------------------------------------------- #
def _config() -> SeverityConfig:
    """默认参数（与 §4.1.3 一致）的解析结果。"""
    return SeverityConfig.from_dict({"enabled": True})


def test_severe_min_threshold_flips_grade():
    # 默认 0.70 下 0.511 → 1；把 severe_min 降到 0.45 → 2
    default = grade([make_detection()], {"enabled": True})[0]
    assert default.severity == SEVERITY_MODERATE
    flipped = grade([make_detection()], {"enabled": True, "grade_thresholds": {"severe_min": 0.45}})[0]
    assert flipped.severity == SEVERITY_SEVERE


def test_mild_max_threshold_flips_grade():
    # score 0.511：默认 mild_max=0.35 → 1；抬高到 0.55 → 0
    assert grade([make_detection()], {"enabled": True})[0].severity == SEVERITY_MODERATE
    raised = grade([make_detection()], {"enabled": True, "grade_thresholds": {"mild_max": 0.55}})[0]
    assert raised.severity == SEVERITY_MILD


def test_cfg_none_or_empty_disables_grading():
    detection = make_detection()
    assert grade([detection], None)[0].severity is None
    assert grade([detection], {})[0].severity is None


def test_enabled_false_disables_grading():
    assert grade([make_detection()], {"enabled": False})[0].severity is None


def test_missing_optional_fields_take_defaults():
    detection = make_detection()
    assert grade([detection], {"enabled": True})[0].severity == SEVERITY_MODERATE


@pytest.mark.parametrize(
    "cfg",
    [
        {"unknown_field": 1},
        {"enabled": "yes"},
        {"weights": {"area": 0.8, "confidence": 0.5}},
        {"weights": {"area": -0.1}},
        {"weights": {"unknown": 0.1}},
        {"area_ratio_ref": {2: 0.03}},
        {"area_ratio_ref": {4: 0.03, "default": 0.02}},
        {"area_ratio_ref": {2: 0.0, "default": 0.02}},
        {"area_ratio_ref": {2: math.nan, "default": 0.02}},
        {"class_bias": {2: 1.5, "default": 0.0}},
        {"class_bias": {"unknown": 0.0, "default": 0.0}},
        {"conf_floor": 0.95, "conf_ceiling": 0.25},
        {"conf_floor": -0.1},
        {"grade_thresholds": {"mild_max": 0.8, "severe_min": 0.4}},
        {"grade_thresholds": {"mild_max": 0.0}},
        {"grade_thresholds": {"severe_min": 1.0}},
        {"grade_thresholds": {"unknown": 0.5}},
    ],
)
def test_invalid_cfg_raises_value_error(cfg):
    with pytest.raises(ValueError):
        grade([make_detection()], cfg)


def test_cfg_must_be_dict():
    with pytest.raises(ValueError, match="dict"):
        grade([make_detection()], "enabled")


# --------------------------------------------------------------------------- #
# 不可变性：返回新对象，不改入参
# --------------------------------------------------------------------------- #
def test_grade_does_not_mutate_input():
    detection = make_detection(severity=None)
    result = grade([detection], {"enabled": True})
    assert detection.severity is None
    assert result[0] is not detection
    assert result[0].severity == SEVERITY_MODERATE
    assert result[0].class_id == detection.class_id


# --------------------------------------------------------------------------- #
# §4.1.4 边界回退
# --------------------------------------------------------------------------- #
def test_nan_and_inf_confidence_grade_to_none(caplog):
    detections = [
        make_detection(confidence=math.nan),
        make_detection(confidence=math.inf),
        make_detection(area_ratio=math.nan),
    ]
    with caplog.at_level("WARNING", logger="defect.severity"):
        result = grade(detections, {"enabled": True})
    assert all(item.severity is None for item in result)
    assert len(caplog.records) == 3


def test_unknown_class_id_grades_to_none(caplog):
    detection = make_detection(class_id=2)
    detection.class_id = 9
    with caplog.at_level("WARNING", logger="defect.severity"):
        result = grade([detection], {"enabled": True})
    assert result[0].severity is None
    assert caplog.records


def test_missing_class_key_falls_back_to_default():
    detection = make_detection(class_id=3)
    result = grade([detection], {"enabled": True, "area_ratio_ref": {"default": 0.02}})
    assert result[0].severity is not None


@pytest.mark.parametrize("bad", [None, "not-a-list", (make_detection(),)])
def test_bad_detections_raise_value_error(bad):
    with pytest.raises(ValueError, match="detections"):
        grade(bad)


def test_non_detection_element_raises_value_error():
    with pytest.raises(ValueError, match="Detection"):
        grade(["not-a-detection"])
