"""缺口链路健壮性（2026-10-27）：凸包顶点处出现重复坐标的轮廓不得导致整帧失败。

背景：`datasets/neu_det/images/val/rolled-in_scale_146.jpg` 的最大轮廓在 `(143, 0)`
出现两次（index 0 与 index 2），该点又是凸包顶点，`cv2.convexHull(returnPoints=False)`
返回非单调索引，`cv2.convexityDefects` 抛 `cv2.error`，被 `detect()` 转成
`DetectorInferenceError` 后整帧失败。修复方式：轮廓保序去重后再算凸性缺陷，
并在 `convexityDefects` 外层兜底跳过病态轮廓。

用 OpenCV 现场合成轮廓，不依赖 `datasets/` 下任何图片（与既有单测同一约定）。
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detectors.opencv_detector import OpencvDetector  # noqa: E402


def _slit_contour() -> np.ndarray:
    """顶边出现非相邻重复坐标（模拟"裂缝"）：凸包索引不单调，OpenCV 直接报错。

    形态与现场 146.jpg 同类：同一坐标在轮廓序列里出现两次，且该点是凸包顶点。
    """
    return np.array(
        [[10, 0], [90, 0], [10, 0], [90, 40], [10, 40]], dtype=np.int32
    ).reshape(-1, 1, 2)


def _backtrack_contour() -> np.ndarray:
    """相邻重复坐标（纯回折）：几何等价，面积/外接框不因去重改变。"""
    return np.array(
        [[10, 0], [90, 0], [90, 0], [90, 40], [10, 40]], dtype=np.int32
    ).reshape(-1, 1, 2)


def _gray_canvas() -> np.ndarray:
    """`_detect_notch` 内部只做 Otsu 阈值 + 形态学，灰度值本身不影响注入的轮廓。"""
    return np.full((200, 200), 128, dtype=np.uint8)


def test_raw_convexity_defects_reproduces_error():
    """前提校验：未去重时 OpenCV 确实报错，保证本用例真的覆盖了该缺陷。"""
    contour = _slit_contour()
    hull = cv2.convexHull(contour, returnPoints=False)
    with pytest.raises(cv2.error):
        cv2.convexityDefects(contour, hull)


def test_dedupe_contour_removes_duplicates_and_keeps_order():
    """去重只删重复顶点：返回 int32、(N,1,2)、保序（保留首次出现的点）。"""
    contour = _slit_contour()
    deduped = OpencvDetector._dedupe_contour(contour)

    assert deduped.dtype == np.int32
    assert deduped.shape[1:] == (1, 2)
    assert len(contour) == 5 and len(deduped) == 4
    assert deduped.reshape(-1, 2)[0].tolist() == [10, 0]
    assert deduped.reshape(-1, 2).tolist() == [[10, 0], [90, 0], [90, 40], [10, 40]]


def test_dedupe_makes_convexity_defects_work_for_slit_contour():
    """主修的直接效果：去重后同一轮廓不再触发 OpenCV 报错。"""
    deduped = OpencvDetector._dedupe_contour(_slit_contour())
    hull = cv2.convexHull(deduped, returnPoints=False)

    defects = cv2.convexityDefects(deduped, hull)  # 修复前这里会抛 cv2.error
    assert defects is None or len(defects) >= 0


def test_dedupe_keeps_geometry_for_adjacent_duplicate():
    """相邻重复属于纯回折：去重前后面积/外接框完全一致（几何不变）。"""
    contour = _backtrack_contour()
    deduped = OpencvDetector._dedupe_contour(contour)

    assert cv2.contourArea(deduped) == pytest.approx(cv2.contourArea(contour))
    assert cv2.boundingRect(deduped) == cv2.boundingRect(contour)


def test_detect_notch_survives_duplicate_hull_vertex(monkeypatch):
    """注入病态轮廓后，缺口链路返回 list 且不抛异常（去重主修生效）。"""
    detector = OpencvDetector()
    monkeypatch.setattr(cv2, "findContours", lambda *args, **kwargs: ([_slit_contour()], None))

    candidates = detector._detect_notch(_gray_canvas())

    assert isinstance(candidates, list)
    for candidate in candidates:
        assert candidate.class_id == 3


def test_detect_notch_falls_back_when_convexity_defects_raises(monkeypatch):
    """兜底：去重后若 OpenCV 仍报错，只跳过该轮廓，返回空列表而不是抛异常。"""

    def _boom(*args, **kwargs):
        raise cv2.error("simulated convexityDefects failure")

    detector = OpencvDetector()
    monkeypatch.setattr(cv2, "findContours", lambda *args, **kwargs: ([_slit_contour()], None))
    monkeypatch.setattr(cv2, "convexityDefects", _boom)

    assert detector._detect_notch(_gray_canvas()) == []
