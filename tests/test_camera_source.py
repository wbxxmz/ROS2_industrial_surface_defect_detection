"""阶段 2 CameraSource 单测：后台线程 + 队列 + 重连 + close 幂等。

不依赖真实摄像头：用 FakeCapture 替换 cv2.VideoCapture，可控地模拟
正常出帧、队列满、断流、重连、尺寸 0 等场景（见阶段 2 §4.3 / §4.4）。

替身默认带 read_delay 节流：否则生产者线程会把消费者远远甩开，
帧号断言失去意义（也避免紧循环空转）。
"""

import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inputs import camera_source as camera_module  # noqa: E402
from inputs.camera_source import CameraSource  # noqa: E402
from utils.exceptions import InputSourceError, SourceOpenError  # noqa: E402

W, H = 8, 6
DELAY = 0.03


class FakeCapture:
    """可控 VideoCapture 替身。

    :param fail_after: 该实例连续出帧 N 次后开始失败（None 表示永不失败）
    :param open_ok: isOpened() 返回值
    :param zero_size_after: 该实例连续出帧 N 次后开始返回尺寸 0 图像
    :param read_delay: 每次 read 的耗时，用于节流生产者
    """

    def __init__(self, fail_after=None, open_ok=True, zero_size_after=None, read_delay=0.0):
        self.fail_after = fail_after
        self.zero_size_after = zero_size_after
        self.read_delay = read_delay
        self._open_ok = open_ok
        self.reads = 0
        self.released = False

    def isOpened(self):  # noqa: N802 - 对齐 OpenCV 接口命名
        return self._open_ok

    def read(self):
        if self.read_delay:
            time.sleep(self.read_delay)
        self.reads += 1
        if self.fail_after is not None and self.reads > self.fail_after:
            return False, None
        if self.zero_size_after is not None and self.reads > self.zero_size_after:
            return True, np.zeros((0, 0, 3), dtype=np.uint8)
        return True, np.full((H, W, 3), self.reads % 256, dtype=np.uint8)

    def set(self, *_args):
        return True

    def get(self, *_args):
        return 0.0

    def release(self):
        self.released = True


@pytest.fixture()
def patch_capture(monkeypatch):
    """把 cv2.VideoCapture 换成可控替身；返回安装函数（返回每次创建的实例列表）。"""

    def install(**kwargs):
        instances = []

        def factory(*_args, **_kw):
            cap = FakeCapture(**kwargs)
            instances.append(cap)
            return cap

        monkeypatch.setattr(camera_module.cv2, "VideoCapture", factory)
        return instances

    return install


@pytest.fixture()
def make_source():
    """创建并 open 一个 CameraSource；用例结束统一 close，避免遗留非 daemon 线程。"""
    created = []

    def _make(**overrides):
        cfg = {"type": "camera", "device_id": 0, "timeout": 0.3}
        cfg.update(overrides)
        source = CameraSource()
        source.open(cfg)
        created.append(source)
        return source

    yield _make
    for source in created:
        source.close()  # 幂等；确保后台线程已退出


# ---------- 正常路径 ----------


def test_camera_reads_frames_with_fields(patch_capture, make_source):
    """正常出帧：字段与 §5.4.4 一致，frame_index 从 0 递增。"""
    patch_capture(open_ok=True, read_delay=DELAY)
    source = make_source()

    frame = source.read()
    assert frame is not None
    assert (frame.width, frame.height) == (W, H)
    assert frame.image.dtype == np.uint8
    assert frame.frame_index == 0
    assert frame.timestamp >= 0.0


def test_camera_read_is_blocking_not_none_when_queue_empty(patch_capture, make_source):
    """队列暂时为空 ≠ 读尽：read 应阻塞等待而非立刻返回 None。"""
    patch_capture(open_ok=True, read_delay=DELAY)
    source = make_source()

    frame = source.read()  # 若误判为空队列即读尽，这里会是 None
    assert frame is not None


def test_camera_step_keeps_source_frame_index(patch_capture, make_source):
    """跳帧：step=3 入队帧的 frame_index 为源帧号，不因跳帧变成产出序号。"""
    patch_capture(open_ok=True, read_delay=DELAY)
    source = make_source(step=3)

    indices = [source.read().frame_index for _ in range(4)]
    assert indices == [0, 3, 6, 9]


# ---------- 队列策略 ----------


def test_camera_queue_full_keeps_latest(patch_capture, make_source):
    """队列满：丢弃最旧帧、保留最新帧（§4.3）。"""
    patch_capture(open_ok=True)  # 不节流：让生产者跑满队列
    source = make_source(queue_size=2)

    time.sleep(0.15)
    frame = source.read()
    assert frame is not None
    # 容量 2 的循环队列，读到的帧号应远大于 1，说明最旧帧已被丢弃
    assert frame.frame_index > 1


# ---------- 断流与重连 ----------


def test_camera_reconnect_exhausted_raises_input_source_error(patch_capture, make_source):
    """断流且重连用尽：read 抛 InputSourceError，且重连次数与上限一致（§4.4）。"""
    instances = patch_capture(open_ok=True, fail_after=0)
    source = make_source(reconnect=True, reconnect_interval=0.01, reconnect_max=2)

    with pytest.raises(InputSourceError):
        for _ in range(20):
            source.read()
    # 初次打开 1 次 + 重连 2 次 = 3
    assert len(instances) == 3


def test_camera_no_reconnect_raises_immediately(patch_capture, make_source):
    """reconnect=false：断流即抛 InputSourceError（队列中已捕获帧先交付）。"""
    patch_capture(open_ok=True, fail_after=1, read_delay=DELAY)
    source = make_source(reconnect=False, reconnect_max=0)

    assert source.read() is not None  # 队列里那一帧先交付
    with pytest.raises(InputSourceError):
        for _ in range(20):
            source.read()


def test_camera_zero_size_frame_raises(patch_capture, make_source):
    """尺寸 0 帧：抛 InputSourceError（§5 read 边界）。"""
    patch_capture(open_ok=True, zero_size_after=0)
    source = make_source(reconnect=False)

    with pytest.raises(InputSourceError):
        for _ in range(20):
            source.read()


# ---------- 打开失败 ----------


def test_camera_open_failure_raises_source_open_error(patch_capture):
    patch_capture(open_ok=False)
    with pytest.raises(SourceOpenError):
        CameraSource().open({"type": "camera", "device_id": 0})


def test_camera_wrong_type_raises_value_error(patch_capture):
    patch_capture(open_ok=True)
    with pytest.raises(ValueError):
        CameraSource().open({"type": "video", "path": "x.mp4"})


def test_camera_read_before_open_raises():
    with pytest.raises(InputSourceError):
        CameraSource().read()


# ---------- 关闭与生命周期 ----------


def test_camera_close_stops_thread_and_is_idempotent(patch_capture):
    """close：线程退出、句柄释放、可重复调用（§4.3 / §5）。"""
    patch_capture(open_ok=True, read_delay=DELAY)
    source = CameraSource()
    source.open({"type": "camera", "device_id": 0, "timeout": 0.3})
    source.read()

    source.close()
    assert source._thread is not None and not source._thread.is_alive()
    assert source._cap is None
    source.close()  # 幂等
    assert source.close() is None


def test_camera_close_without_read_does_not_hang(patch_capture):
    """未 read 即 close：不得挂起（重连等待需可被 close 打断）。"""
    patch_capture(open_ok=True, fail_after=0)
    source = CameraSource()
    source.open(
        {
            "type": "camera",
            "device_id": 0,
            "timeout": 0.3,
            "reconnect": True,
            "reconnect_interval": 5.0,
            "reconnect_max": 3,
        }
    )

    done = threading.Event()

    def _close():
        source.close()
        done.set()

    threading.Thread(target=_close, daemon=True).start()
    assert done.wait(2.0), "close() 在重连等待期间未能及时返回"
    assert not source._thread.is_alive()


def test_camera_read_after_close_returns_none(patch_capture):
    """close 后 read 返回 None（读尽语义一致）。"""
    patch_capture(open_ok=True, read_delay=DELAY)
    source = CameraSource()
    source.open({"type": "camera", "device_id": 0, "timeout": 0.3})
    source.close()
    assert source.read() is None


def test_camera_reopen_restarts_cleanly(patch_capture):
    """同一实例 open 两次：先关旧线程，再起新线程。"""
    patch_capture(open_ok=True, read_delay=DELAY)
    source = CameraSource()
    source.open({"type": "camera", "device_id": 0, "timeout": 0.3})
    first_thread = source._thread

    source.open({"type": "camera", "device_id": 0, "timeout": 0.3})
    assert source._thread is not first_thread
    assert not first_thread.is_alive()
    source.close()
