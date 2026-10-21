"""阶段 5 §7 验收 #3/#4/#5：判定规则配置化、PASS/FAIL 语义、统计汇总一致性。"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import Detection, Verdict  # noqa: E402
from detectors.opencv_detector import CLASS_MAP  # noqa: E402
from judge import judge, summarize  # noqa: E402


def make_detection(
    class_id: int = 0,
    confidence: float = 0.90,
    area_px: int = 100,
    area_ratio: float = 0.002,
    severity=None,
) -> Detection:
    """构造符合阶段 0 契约的 Detection（bbox 固定，不影响判定）。"""
    return Detection(
        class_id=class_id,
        class_name=CLASS_MAP[class_id],
        bbox=(0, 0, 10, 10),
        area_px=area_px,
        area_ratio=area_ratio,
        confidence=confidence,
        mask=None,
        severity=severity,
    )


# --------------------------------------------------------------------------- #
# §7.4 PASS / FAIL 语义
# --------------------------------------------------------------------------- #
def test_empty_detections_returns_pass():
    verdict = judge([])
    assert verdict.is_pass is True
    assert verdict.reason.startswith("PASS")
    assert verdict.total_count == 0
    assert verdict.triggered == []
    assert verdict.total_area_px == 0
    assert verdict.total_area_ratio == 0.0


def test_empty_detections_class_counts_have_four_keys():
    verdict = judge([])
    assert set(verdict.class_counts) == {0, 1, 2, 3}
    assert all(value == 0 for value in verdict.class_counts.values())


def test_fail_reason_and_triggered_format():
    verdict = judge([make_detection()])
    assert verdict.is_pass is False
    assert verdict.reason.startswith("FAIL")
    assert verdict.triggered == ["count:1>0"]
    assert verdict.reason == "FAIL：count:1>0"


def test_pass_reason_when_all_detections_filtered_out():
    # confidence 低于 min_confidence 默认值 0.45：不算有效缺陷 → PASS
    verdict = judge([make_detection(confidence=0.40)])
    assert verdict.is_pass is True
    assert verdict.total_count == 0
    assert verdict.reason == "PASS：未检出有效缺陷"


def test_pass_reason_when_rules_not_triggered():
    verdict = judge([make_detection()], {"max_count": 1})
    assert verdict.is_pass is True
    assert verdict.total_count == 1
    assert verdict.reason == "PASS：未触发判定规则"


# --------------------------------------------------------------------------- #
# §7.3 配置化与阈值翻转
# --------------------------------------------------------------------------- #
def test_max_count_threshold_flips_result():
    detections = [make_detection()]
    fail = judge(detections, {"max_count": 0})
    pass_ = judge(detections, {"max_count": 1})
    assert fail.is_pass is False and fail.triggered == ["count:1>0"]
    assert pass_.is_pass is True and pass_.triggered == []


def test_min_confidence_is_configurable():
    detections = [make_detection(confidence=0.40)]
    assert judge(detections).total_count == 0  # 默认 0.45 过滤掉
    assert judge(detections, {"min_confidence": 0.30}).total_count == 1


def test_min_area_px_is_configurable():
    detections = [make_detection(area_px=5)]
    assert judge(detections).total_count == 0  # 默认 10 过滤掉
    assert judge(detections, {"min_area_px": 0}).total_count == 1


def test_area_ratio_rule_triggers():
    detections = [make_detection(area_ratio=0.02)]
    verdict = judge(detections, {"max_count": 10})
    assert verdict.is_pass is False
    assert verdict.triggered == ["area:0.0200>0.01"]


def test_area_ratio_sum_is_clipped_before_comparison():
    detections = [make_detection(area_ratio=0.6), make_detection(area_ratio=0.6)]
    verdict = judge(detections, {"max_count": 10, "max_area_ratio": 0.5})
    assert verdict.total_area_ratio == 1.0  # 1.2 → clip 到 1.0
    assert verdict.triggered == ["area:1.0000>0.5"]


def test_count_rule_precedes_area_rule_in_triggered():
    detections = [make_detection(area_ratio=0.5)]
    verdict = judge(detections)
    assert verdict.triggered == ["count:1>0", "area:0.5000>0.01"]


def test_per_class_and_global_rules_both_triggered():
    detections = [make_detection(class_id=0) for _ in range(3)]
    verdict = judge(
        detections,
        {"max_count": 0, "per_class": {0: {"max_count": 2}}},
    )
    assert verdict.is_pass is False
    assert verdict.triggered == ["count:3>0", "class:count:0:3>2"]
    assert verdict.class_counts == {0: 3, 1: 0, 2: 0, 3: 0}


def test_per_class_does_not_replace_global_rule():
    # 全局宽松（5）、类内严格（1）：只有类内规则命中
    detections = [make_detection(class_id=0), make_detection(class_id=0)]
    verdict = judge(
        detections,
        {"max_count": 5, "per_class": {0: {"max_count": 1}}},
    )
    assert verdict.is_pass is False
    assert verdict.triggered == ["class:count:0:2>1"]


def test_per_class_area_rule_triggers():
    detections = [make_detection(class_id=2, area_ratio=0.3)]
    verdict = judge(
        detections,
        {"max_count": 10, "max_area_ratio": 0.5, "per_class": {"stain": {"max_area_ratio": 0.1}}},
    )
    assert verdict.triggered == ["class:area:2:0.3000>0.1"]


def test_per_class_rejects_duplicate_class_key():
    with pytest.raises(ValueError, match="重复配置"):
        judge([], {"per_class": {0: {}, "scratch": {}}})


@pytest.mark.parametrize(
    "cfg",
    [
        {"unknown_field": 1},
        {"max_count": -1},
        {"max_count": True},
        {"max_count": 1.5},
        {"max_area_ratio": 1.5},
        {"min_confidence": -0.1},
        {"min_area_px": -1},
        {"max_severity": "严重"},
        {"per_class": {4: {"max_count": 1}}},
        {"per_class": {"unknown": {"max_count": 1}}},
        {"per_class": {0: {"unknown": 1}}},
        {"per_class": []},
        {"per_class": {0: 5}},
    ],
)
def test_invalid_cfg_raises_value_error(cfg):
    with pytest.raises(ValueError):
        judge([], cfg)


def test_cfg_none_uses_documented_defaults():
    verdict = judge([make_detection(confidence=0.44)], None)
    assert verdict.is_pass is True  # 默认 min_confidence=0.45 过滤
    assert verdict.total_count == 0


def test_severity_does_not_trigger_in_stage5():
    # 阶段 5 max_severity 恒为 null，severity 有值也不参与判定
    verdict = judge([make_detection(severity="严重")], {"max_count": 10})
    assert verdict.is_pass is True


def test_unknown_class_id_is_excluded():
    detection = make_detection()
    detection.class_id = 9
    detection.class_name = "unknown"
    verdict = judge([detection])
    assert verdict.total_count == 0
    assert sum(verdict.class_counts.values()) == 0


# --------------------------------------------------------------------------- #
# §5.1 输入校验
# --------------------------------------------------------------------------- #
def test_none_detections_raises_value_error():
    with pytest.raises(ValueError, match="detections"):
        judge(None)


def test_non_detection_element_raises_value_error():
    with pytest.raises(ValueError, match="not|不是"):
        judge(["not-a-detection"])


# --------------------------------------------------------------------------- #
# §7.5 统计汇总
# --------------------------------------------------------------------------- #
def _verdict(is_pass: bool, counts: dict, area_px: int, area_ratio: float) -> Verdict:
    return Verdict(
        is_pass=is_pass,
        reason="PASS：未检出有效缺陷" if is_pass else "FAIL：count:1>0",
        total_count=sum(counts.values()),
        class_counts=counts,
        total_area_px=area_px,
        total_area_ratio=area_ratio,
        triggered=[] if is_pass else ["count:1>0"],
    )


def test_summarize_empty_batch():
    summary = summarize([])
    assert summary["sample_count"] == 0
    assert summary["pass_count"] == 0
    assert summary["fail_count"] == 0
    assert summary["yield_rate"] == 0.0
    assert summary["total_count"] == 0
    assert summary["class_counts"] == {0: 0, 1: 0, 2: 0, 3: 0}


def test_summarize_matches_source_verdicts():
    verdicts = [
        _verdict(True, {0: 0, 1: 0, 2: 0, 3: 0}, 0, 0.0),
        _verdict(True, {0: 1, 1: 0, 2: 0, 3: 0}, 100, 0.002),
        _verdict(False, {0: 2, 1: 1, 2: 0, 3: 0}, 300, 0.006),
    ]
    summary = summarize(verdicts)
    assert summary["sample_count"] == 3
    assert summary["pass_count"] == 2
    assert summary["fail_count"] == 1
    assert summary["yield_rate"] == pytest.approx(2 / 3)
    assert summary["total_count"] == 4
    # §7.5 恒等式：缺陷总数 == 各类别计数之和 == 各样本 total_count 之和
    assert summary["total_count"] == sum(summary["class_counts"].values())
    assert summary["total_count"] == sum(v.total_count for v in verdicts)
    assert summary["class_counts"] == {0: 3, 1: 1, 2: 0, 3: 0}
    assert summary["total_area_px"] == 400
    assert summary["total_area_ratio"] == pytest.approx(0.008)


def test_summarize_clips_area_ratio():
    verdicts = [
        _verdict(False, {0: 1}, 100, 0.8),
        _verdict(False, {0: 1}, 100, 0.8),
    ]
    assert summarize(verdicts)["total_area_ratio"] == 1.0


def test_summarize_rejects_invalid_input():
    with pytest.raises(ValueError):
        summarize(None)
    with pytest.raises(ValueError):
        summarize(["not-a-verdict"])
