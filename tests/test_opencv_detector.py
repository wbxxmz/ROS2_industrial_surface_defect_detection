"""阶段 3 §7：OpencvDetector 单元测试（契约字段 / 异常语义 / 边界过滤 / 置信度范围 / 映射表一致）。

测试图画用 OpenCV 现场合成，不依赖仓库内二进制资源（与阶段 2 单测同一约定）。
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import Detection, Frame  # noqa: E402
from detectors.opencv_detector import CLASS_MAP, MAX_BBOX_AREA_RATIO, OpencvDetector  # noqa: E402
from utils.exceptions import DetectorLoadError  # noqa: E402

FRAME_W, FRAME_H = 200, 200
IMAGE_AREA = FRAME_W * FRAME_H


# ---------- 素材构造 ----------


def _make_frame(image: np.ndarray, frame_index: int = 0) -> Frame:
    height, width = int(image.shape[0]), int(image.shape[1])
    return Frame(image=image, width=width, height=height, frame_index=frame_index, timestamp=0.0)


def _detector(params: dict | None = None) -> OpencvDetector:
    """按 §6 契约路径构造：load({"params": ...})，缺省用 §4.4 默认值。"""
    detector = OpencvDetector()
    assert detector.load({"params": params or {}}) is None
    return detector


def _defect_frame() -> Frame:
    """灰底 + 深色圆斑（凹坑）+ 深色粗线（划痕）：保证至少一类被检出。"""
    image = np.full((FRAME_H, FRAME_W, 3), 160, dtype=np.uint8)
    cv2.circle(image, (60, 60), 8, (20, 20, 20), -1)
    cv2.line(image, (30, 150), (170, 150), (20, 20, 20), 3)
    return _make_frame(image)


def _pacman_frame(radius: int) -> Frame:
    """亮色圆盘缺一角（pac-man）；缺口三角形外接框随半径平方增长。"""
    image = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    cv2.ellipse(image, (100, 100), (radius, radius), 0, 90, 360, (255, 255, 255), -1)
    return _make_frame(image)


def _blank_frame() -> Frame:
    return _make_frame(np.full((100, 100, 3), 128, dtype=np.uint8))


# ---------- §7-1 契约字段完整性 ----------


def test_detect_returns_list_of_detection_with_contract_fields():
    """detect 返回 list[Detection]，且 bbox / area_px / area_ratio / confidence 类型与范围合规。"""
    detections = _detector().detect(_defect_frame())

    assert isinstance(detections, list)
    assert len(detections) >= 1, "合成缺陷图应至少检出 1 条（否则本用例为空跑）"

    for detection in detections:
        assert isinstance(detection, Detection)
        # class_id / class_name 必须同属 §4.1 映射表
        assert isinstance(detection.class_id, int) and not isinstance(detection.class_id, bool)
        assert detection.class_id in CLASS_MAP
        assert detection.class_name == CLASS_MAP[detection.class_id]
        # bbox：像素坐标 (x, y, w, h)，原点左上，不越界，非归一化
        assert isinstance(detection.bbox, tuple) and len(detection.bbox) == 4
        x, y, w, h = detection.bbox
        assert all(isinstance(value, int) and value >= 0 for value in (x, y, w, h))
        assert w >= 1 and h >= 1
        assert x + w <= FRAME_W and y + h <= FRAME_H
        # area_px：缺陷实际像素面积，≠ bbox 矩形面积
        assert isinstance(detection.area_px, int) and detection.area_px >= 1
        assert detection.area_px <= w * h
        assert detection.area_px >= 10  # §4.5 默认 min_area 下限
        # area_ratio：0~1，且与 area_px / 整图面积 口径一致
        assert isinstance(detection.area_ratio, float)
        assert 0.0 < detection.area_ratio <= 1.0
        assert detection.area_ratio == pytest.approx(detection.area_px / IMAGE_AREA)
        # 掩码：uint8、0/255、尺寸等于帧图像
        assert detection.mask is not None and detection.mask.dtype == np.uint8
        assert detection.mask.shape == (FRAME_H, FRAME_W)
        assert set(np.unique(detection.mask)) <= {0, 255}
        # 阶段 6 前 severity 恒为 None
        assert detection.severity is None


def test_detector_contract_members():
    """name / version / release 契约成员稳定且幂等（§6）。"""
    detector = OpencvDetector()
    assert detector.name == "opencv"
    assert detector.version == "0.1.0"
    assert detector.release() is None
    assert detector.release() is None


# ---------- §7-4 置信度范围 ----------


def test_confidence_is_never_none_and_within_range():
    """放宽 min_confidence 后取全量结果，逐条确认 confidence 恒非 None 且落在 [0, 1]。"""
    detections = _detector({"min_confidence": 0.0, "min_area": 1}).detect(_defect_frame())
    assert detections, "放宽阈值后应能拿到候选，否则无法验证置信度范围"
    for detection in detections:
        assert detection.confidence is not None
        assert isinstance(detection.confidence, float)
        assert 0.0 <= detection.confidence <= 1.0


# ---------- §7-2 异常语义 ----------


def test_detect_none_frame_raises_value_error():
    with pytest.raises(ValueError):
        _detector().detect(None)


@pytest.mark.parametrize("shape", [(0, 0, 3), (0, FRAME_W, 3), (FRAME_H, 0, 3)])
def test_detect_zero_size_frame_raises_value_error(shape):
    """尺寸为 0（含空图像）必须抛 ValueError，而不是返回 [] 或抛别的异常。"""
    frame = Frame(image=np.zeros(shape, dtype=np.uint8), width=0, height=0, frame_index=0, timestamp=0.0)
    with pytest.raises(ValueError):
        _detector().detect(frame)


def test_detect_size_mismatch_frame_raises_value_error():
    frame = Frame(
        image=np.zeros((10, 10, 3), dtype=np.uint8), width=99, height=10, frame_index=0, timestamp=0.0
    )
    with pytest.raises(ValueError):
        _detector().detect(frame)


def test_detect_blank_frame_returns_empty_list():
    """无缺陷帧返回空列表（不是 None，也不抛异常）。"""
    detections = _detector().detect(_blank_frame())
    assert detections == []


def test_load_requires_params_key():
    with pytest.raises(DetectorLoadError):
        OpencvDetector().load({"type": "opencv"})


@pytest.mark.parametrize(
    "params",
    [
        {"denoise_kernel": 4},  # 偶数核（核大小必须为奇数）
        {"denoise": "bilateral"},  # 不支持的枚举值
        {"min_area": 0},  # 下限越界
        {"min_confidence": 1.5},  # 超出 [0, 1]
        {"scratch_canny_low": 200, "scratch_canny_high": 100},  # 低阈值 > 高阈值
        {"dent_radius_min": 30, "dent_radius_max": 10},  # 半径区间倒挂
        {"unknown_param": 1},  # 未知字段
    ],
)
def test_load_rejects_invalid_params(params):
    with pytest.raises(DetectorLoadError):
        OpencvDetector().load({"params": params})


# ---------- §7-3 边界过滤 ----------


def test_min_area_filter_drops_small_candidates():
    """min_area 严格生效：恰好等于下限的候选保留，低于下限的全部丢弃。"""
    frame = _defect_frame()
    detections = _detector().detect(frame)
    assert detections, "默认 min_area=10 下应有检出"

    threshold = min(detection.area_px for detection in detections)

    kept = _detector({"min_area": threshold}).detect(frame)
    assert any(detection.area_px == threshold for detection in kept), "等于下限的候选应保留"

    strict = _detector({"min_area": threshold + 1}).detect(frame)
    assert all(detection.area_px >= threshold + 1 for detection in strict)
    assert len(strict) < len(kept), "抬高下限后应少检出一条"

    # 下限高过任何轮廓面积 → 全部丢弃
    assert _detector({"min_area": IMAGE_AREA + 1}).detect(frame) == []


def test_oversize_bbox_filter_drops_large_bbox():
    """§4.5 硬性过滤：bbox 矩形面积超过整图 15% 的检测必须被丢弃。"""
    assert MAX_BBOX_AREA_RATIO == 0.15
    limit = MAX_BBOX_AREA_RATIO * IMAGE_AREA

    # 对照组：小尺寸缺口（bbox 占比 ≈ 9%）应被正常检出
    small = _detector({"min_confidence": 0.0}).detect(_pacman_frame(60))
    assert [d for d in small if d.class_id == 3], "对照组：小尺寸缺口应被检出且未被面积上限误杀"
    assert all(d.bbox[2] * d.bbox[3] < limit for d in small)

    # 实验组：同构图形放大到 bbox 占比 ≈ 20%，缺口候选应被硬性过滤掉
    large = _detector({"min_confidence": 0.0}).detect(_pacman_frame(90))
    assert all(d.bbox[2] * d.bbox[3] <= limit for d in large)
    assert not [d for d in large if d.class_id == 3], "bbox 超 15% 的缺口应被硬性过滤"


# ---------- §7-5 映射表一致性 ----------


def test_class_map_is_frozen():
    """§4.1 映射表写死，阶段 4/5/7 复用，不得被改动。"""
    assert CLASS_MAP == {0: "scratch", 1: "dent", 2: "stain", 3: "notch"}
