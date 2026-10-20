"""阶段 2 输入源单测：ImageSource / VideoSource（含契约一致性、边界、幂等）。

测试用图片/视频在 tmp_path 内用 OpenCV 现场生成，不依赖仓库内二进制资源。
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inputs.image_source import ImageSource  # noqa: E402
from inputs.video_source import VideoSource  # noqa: E402
from utils.exceptions import InputSourceError, ProjectError, SourceOpenError  # noqa: E402

FRAME_W, FRAME_H = 16, 12


# ---------- 素材构造 ----------


def _write_image(path: Path, value: int = 128) -> None:
    image = np.full((FRAME_H, FRAME_W, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image), f"测试图片写入失败：{path}"


def _write_video(path: Path, frames: int, fps: int = 25) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (FRAME_W, FRAME_H)
    )
    assert writer.isOpened(), f"测试视频写入器打开失败：{path}"
    for i in range(frames):
        writer.write(np.full((FRAME_H, FRAME_W, 3), 10 * i % 256, dtype=np.uint8))
    writer.release()


@pytest.fixture()
def single_image(tmp_path: Path) -> Path:
    path = tmp_path / "one.png"
    _write_image(path)
    return path


@pytest.fixture()
def image_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "batch"
    directory.mkdir()
    for name in ("03_c.png", "01_a.png", "02_b.png"):
        _write_image(directory / name)
    (directory / "readme.txt").write_text("非图片，应被扩展名白名单过滤", encoding="utf-8")
    return directory


@pytest.fixture()
def video_file(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp4"
    _write_video(path, frames=7)
    return path


# ---------- ImageSource ----------


def test_image_single_frame_fields(single_image: Path):
    """单张图片：字段与阶段 0 §5.4.4 冻结表一致，读完返回 None。"""
    source = ImageSource()
    source.open({"type": "image", "path": str(single_image)})

    frame = source.read()
    assert frame is not None
    assert frame.image.dtype == np.uint8
    assert frame.image.ndim == 3 and frame.image.shape[2] == 3
    assert (frame.width, frame.height) == (FRAME_W, FRAME_H)
    assert frame.width == frame.image.shape[1]
    assert frame.height == frame.image.shape[0]
    assert frame.frame_index == 0
    assert frame.timestamp == 0.0  # 图片恒为 0.0
    assert source.read() is None  # 读尽
    assert source.read() is None  # 读尽后再读仍为 None，不抛异常


def test_image_dir_sorted_and_no_duplicate(image_dir: Path):
    """目录批量：按文件名排序、无重复无遗漏、frame_index 从 0 递增。"""
    source = ImageSource()
    source.open({"type": "image", "dir": str(image_dir)})

    indices = []
    while (frame := source.read()) is not None:
        indices.append(frame.frame_index)
    assert indices == [0, 1, 2]


def test_image_close_is_idempotent(single_image: Path):
    """close() 幂等：重复调用不抛异常；close 后 read 返回 None。"""
    source = ImageSource()
    source.open({"type": "image", "path": str(single_image)})
    source.close()
    source.close()
    assert source.read() is None


def test_image_read_before_open_raises():
    with pytest.raises(InputSourceError):
        ImageSource().read()


def test_image_missing_file_raises_source_open_error(tmp_path: Path):
    with pytest.raises(SourceOpenError):
        ImageSource().open({"type": "image", "path": str(tmp_path / "nope.png")})


def test_image_corrupt_file_raises_source_open_error(tmp_path: Path):
    """损坏图片在 open 阶段即报 SourceOpenError（§5 open 边界）。"""
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not a real png")
    with pytest.raises(SourceOpenError):
        ImageSource().open({"type": "image", "path": str(broken)})


def test_image_empty_dir_raises_source_open_error(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SourceOpenError):
        ImageSource().open({"type": "image", "dir": str(empty)})


@pytest.mark.parametrize(
    "cfg",
    [
        {"type": "image", "path": "x.png", "step": 0},  # ≤0
        {"type": "image", "path": "x.png", "queue_size": 0},  # ≤0
        {"type": "image"},  # path/dir 二者皆无
        {"type": "image", "path": "a.png", "dir": "b"},  # 二者都有
        {"type": "image", "path": "a.png", "extensionz": [".png"]},  # 未知字段
        {"type": "video", "path": "a.mp4", "start_frame": 5, "end_frame": 2},  # 区间反了
        {"type": "camera"},  # device_id/url 二者皆无
        {"type": "camera", "device_id": 0, "url": "rtsp://x"},  # 二者都有
        {"type": "unknown", "path": "a.png"},  # 未知 type
    ],
)
def test_config_invalid_params_raise_value_error(cfg: dict):
    """非法参数在 open 阶段抛 ValueError（§7 非法参数校验）。"""
    with pytest.raises(ValueError):
        ImageSource().open(cfg)


def test_config_error_is_project_error_subclass_unrelated():
    """契约异常必须继承 ProjectError（便于上层统一捕获）。"""
    assert issubclass(SourceOpenError, ProjectError)
    assert issubclass(InputSourceError, ProjectError)
    assert not issubclass(ValueError, ProjectError)


# ---------- VideoSource ----------


def test_video_step_1_all_frames(video_file: Path):
    """step=1：逐帧产出，frame_index 为源帧号，读完返回 None。"""
    source = VideoSource()
    source.open({"type": "video", "path": str(video_file)})
    assert source._fps == pytest.approx(25.0)

    indices = []
    while (frame := source.read()) is not None:
        indices.append(frame.frame_index)
        assert frame.timestamp == pytest.approx(frame.frame_index / 25.0)
    assert indices == list(range(7))
    source.close()


def test_video_step_3_keeps_source_frame_index(video_file: Path):
    """跳帧：step=3 产出源帧号 0,3,6（frame_index 不变为产出序号）。"""
    source = VideoSource()
    source.open({"type": "video", "path": str(video_file), "step": 3})

    indices = []
    while (frame := source.read()) is not None:
        indices.append(frame.frame_index)
    assert indices == [0, 3, 6]
    source.close()


def test_video_single_frame_file(tmp_path: Path):
    """单帧视频：第一次 read 返回该帧，第二次返回 None。"""
    path = tmp_path / "single.mp4"
    _write_video(path, frames=1)
    source = VideoSource()
    source.open({"type": "video", "path": str(path)})

    assert source.read().frame_index == 0
    assert source.read() is None
    source.close()


def test_video_end_frame_range(video_file: Path):
    """区间：end_frame 为含端点，超出即读尽。"""
    source = VideoSource()
    source.open({"type": "video", "path": str(video_file), "start_frame": 2, "end_frame": 4})

    indices = []
    while (frame := source.read()) is not None:
        indices.append(frame.frame_index)
    assert indices == [2, 3, 4]
    source.close()


def test_video_loop_rewinds(video_file: Path):
    """loop=true：到区间末尾回到起点，不返回 None。"""
    source = VideoSource()
    source.open(
        {"type": "video", "path": str(video_file), "loop": True, "start_frame": 0, "end_frame": 1}
    )

    indices = [source.read().frame_index for _ in range(6)]
    assert indices == [0, 1, 0, 1, 0, 1]
    source.close()


def test_video_close_is_idempotent_and_read_after_close(video_file: Path):
    source = VideoSource()
    source.open({"type": "video", "path": str(video_file)})
    assert source.read() is not None
    source.close()
    source.close()
    assert source.read() is None


def test_video_read_before_open_raises():
    with pytest.raises(InputSourceError):
        VideoSource().read()


def test_video_missing_file_raises_source_open_error(tmp_path: Path):
    with pytest.raises(SourceOpenError):
        VideoSource().open({"type": "video", "path": str(tmp_path / "nope.mp4")})


def test_video_rejects_wrong_type(video_file: Path):
    with pytest.raises(ValueError):
        VideoSource().open({"type": "image", "path": str(video_file)})
