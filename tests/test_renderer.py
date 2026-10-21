"""阶段 5 §7 验收 #1/#2/#4：无头契约、像素级配色/线宽、PASS/FAIL 横幅。"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import Detection, Frame  # noqa: E402
from detectors.opencv_detector import CLASS_MAP  # noqa: E402
from judge import judge  # noqa: E402
from visualization.renderer import (  # noqa: E402
    COLORS,
    _banner_scale,
    _font_scale,
    _thickness,
    render,
    render_to_file,
)


def make_frame(width: int = 200, height: int = 200) -> Frame:
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    return Frame(image=image, width=width, height=height, frame_index=0, timestamp=0.0)


def make_detection(class_id: int = 0, bbox=(100, 100, 50, 50), confidence: float = 0.9, mask=None) -> Detection:
    return Detection(
        class_id=class_id,
        class_name=CLASS_MAP[class_id],
        bbox=bbox,
        area_px=2500,
        area_ratio=0.01,
        confidence=confidence,
        mask=mask,
        severity=None,
    )


def pass_verdict():
    return judge([])


def fail_verdict(detections=None):
    return judge(detections if detections is not None else [make_detection()])


def has_color(image, color, y_slice=slice(0, 60), x_slice=slice(0, 200)) -> bool:
    region = image[y_slice, x_slice]
    return bool(np.any(np.all(region == np.asarray(color, dtype=np.uint8), axis=-1)))


# --------------------------------------------------------------------------- #
# §7.1 无头契约
# --------------------------------------------------------------------------- #
def test_render_returns_same_shape_uint8_and_does_not_mutate_input():
    frame = make_frame()
    original = frame.image.copy()
    image = render(frame, [], pass_verdict())
    assert isinstance(image, np.ndarray)
    assert image.shape == frame.image.shape
    assert image.dtype == np.uint8
    assert np.array_equal(frame.image, original)  # 不修改入参


def test_empty_detections_still_draws_pass_banner():
    image = render(make_frame(), [], pass_verdict())
    assert has_color(image, (0, 200, 0), slice(0, 60), slice(0, 120))


def test_banner_follows_verdict_not_detection_count():
    # 空检测 + FAIL 判定：以 verdict 为准画红色 FAIL（§5.2）
    image = render(make_frame(), [], judge([make_detection()]))
    assert not has_color(image, (0, 200, 0), slice(0, 60), slice(0, 120))
    assert has_color(image, (0, 0, 255), slice(0, 60), slice(0, 120))


def test_no_window_api_is_called(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("无头架构禁止调用 cv2.imshow")

    monkeypatch.setattr(cv2, "imshow", boom)
    monkeypatch.setattr(cv2, "namedWindow", boom)
    assert render(make_frame(), [make_detection()], fail_verdict()).shape == (200, 200, 3)


# --------------------------------------------------------------------------- #
# §7.2 配色与线宽（像素级断言，容差 0）
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("class_id,expected", sorted(COLORS.items()))
def test_box_edge_color_is_exact_bgr(class_id, expected):
    frame = make_frame(400, 400)
    image = render(frame, [make_detection(class_id)], fail_verdict())
    # 左边框中点：x=100, y=125（避开顶部标签与左上角横幅）
    assert tuple(int(v) for v in image[125, 100]) == expected


def test_thickness_scales_with_resolution():
    assert _thickness(200, 200) == 1
    assert _thickness(2000, 2000) == 5


def test_font_scale_scales_with_resolution():
    assert _font_scale(200, 200) == pytest.approx(0.4)
    assert _font_scale(2000, 2000) == pytest.approx(2.0)
    assert _banner_scale(200, 200) == pytest.approx(0.8)
    assert _banner_scale(2000, 2000) == pytest.approx(4.0)


def test_thick_frame_band_at_high_resolution():
    frame = make_frame(2000, 2000)
    detection = make_detection(0, bbox=(100, 100, 100, 100))
    image = render(frame, [detection], fail_verdict([detection]))
    # thickness=5：边框内侧 4 px 仍为红色
    assert tuple(int(v) for v in image[150, 103]) == COLORS[0]


# --------------------------------------------------------------------------- #
# §4.1 / §5.2 边界：clamp 与跳过
# --------------------------------------------------------------------------- #
def test_partially_out_of_bounds_bbox_is_clamped():
    frame = make_frame(100, 100)
    detection = make_detection(0, bbox=(-10, 50, 60, 60))
    image = render(frame, [detection], fail_verdict([detection]))
    assert tuple(int(v) for v in image[80, 0]) == COLORS[0]  # 左边界被 clamp 到 x=0


def test_fully_out_of_bounds_bbox_is_skipped():
    frame = make_frame(100, 100)
    detection = make_detection(0, bbox=(500, 500, 10, 10))
    image = render(frame, [detection], fail_verdict([detection]))
    assert tuple(int(v) for v in image[99, 99]) == (255, 255, 255)


def test_degenerate_bbox_is_skipped():
    frame = make_frame(100, 100)
    detection = make_detection(0, bbox=(10, 10, 0, 10))
    image = render(frame, [detection], fail_verdict([detection]))
    assert image.shape == (100, 100, 3)


def test_mask_overlay_and_shape_mismatch_warning():
    frame = make_frame(200, 200)
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[50:100, 50:100] = 1
    detection = make_detection(0, bbox=(50, 50, 50, 50), mask=mask)
    image = render(frame, [detection], fail_verdict([detection]))
    assert int(image[70, 70, 0]) < 200  # 掩码区域被红色半透明填充

    bad_mask = np.zeros((10, 10), dtype=np.uint8)
    detection.mask = bad_mask
    assert render(frame, [detection], fail_verdict([detection])).shape == (200, 200, 3)


# --------------------------------------------------------------------------- #
# §5.1 入参校验与落盘
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad_frame", [None, "not-a-frame"])
def test_render_rejects_bad_frame(bad_frame):
    with pytest.raises(ValueError, match="frame"):
        render(bad_frame, [], pass_verdict())


def test_render_rejects_zero_size_image():
    frame = Frame(image=np.zeros((0, 0, 3), dtype=np.uint8), width=0, height=0, frame_index=0, timestamp=0.0)
    with pytest.raises(ValueError, match="空|size"):
        render(frame, [], pass_verdict())


def test_render_rejects_non_bgr_image():
    frame = Frame(image=np.zeros((10, 10), dtype=np.uint8), width=10, height=10, frame_index=0, timestamp=0.0)
    with pytest.raises(ValueError, match="H×W×3"):
        render(frame, [], pass_verdict())


def test_render_rejects_size_mismatch():
    frame = make_frame(100, 100)
    frame.width = 200
    with pytest.raises(ValueError, match="宽高"):
        render(frame, [], pass_verdict())


def test_render_rejects_none_detections():
    with pytest.raises(ValueError, match="detections"):
        render(make_frame(), None, pass_verdict())


def test_render_rejects_non_verdict():
    with pytest.raises(ValueError, match="verdict"):
        render(make_frame(), [], {"is_pass": True})


def test_render_to_file_creates_parent_dirs_and_returns_path(tmp_path):
    output = tmp_path / "nested" / "deep" / "frame.jpg"
    written = render_to_file(make_frame(), [make_detection()], fail_verdict(), str(output))
    assert written == str(output)
    assert output.is_file()
    assert cv2.imread(written) is not None


def test_render_to_file_rejects_empty_path():
    with pytest.raises(ValueError, match="output_path"):
        render_to_file(make_frame(), [], pass_verdict(), "")
