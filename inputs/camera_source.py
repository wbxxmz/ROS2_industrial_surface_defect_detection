"""阶段 2 §4.1 / §4.3：CameraSource（USB / RTSP，后台线程 + 帧队列）。

并发策略（阶段 2 §4.3 定稿）：
- 生产者线程持续读帧入队，队列满时丢弃最旧帧（drop-oldest），保留最新帧。
- ``read()`` 阻塞取帧；只有在「线程已退出且队列耗尽」时才返回 None（读尽）。
- 流中断按 §4.4 重连：间隔 ``reconnect_interval``、上限 ``reconnect_max``，用尽抛 InputSourceError。
- ``close()`` 置停止标志 → 等待线程退出 → 释放句柄；幂等。

契约（阶段 0 §5.4.3）：open 失败抛 SourceOpenError；read 出错抛 InputSourceError；close 返回 None。
"""

import queue
import threading
import time
from typing import Optional

import cv2
import numpy as np

from core.input_source import InputSource
from core.types import Frame
from inputs.config import SourceConfig
from utils.exceptions import InputSourceError, SourceOpenError

THREAD_NAME = "CameraSource"


class CameraSource(InputSource):
    """USB 摄像头 / RTSP 网络流输入源。"""

    def __init__(self) -> None:
        self._cfg: Optional[SourceConfig] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._queue: Optional[queue.Queue] = None
        self._stop: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[Exception] = None
        self._start_time = 0.0
        self._opened = False
        self._closed = False

    def open(self, source_cfg: dict) -> None:
        """打开摄像头/网络流并启动后台读帧线程。"""
        cfg = SourceConfig.from_dict(source_cfg)
        if cfg.type != "camera":
            raise ValueError(f"CameraSource 只接受 type='camera'，收到 {cfg.type!r}")

        if self._opened and not self._closed:
            self.close()

        cap = self._build_capture(cfg)
        if not cap.isOpened():
            cap.release()
            raise SourceOpenError(f"摄像头/RTSP 打开失败：{self._describe(cfg)}")

        self._cfg = cfg
        self._cap = cap
        self._queue = queue.Queue(maxsize=cfg.queue_size)
        self._stop = threading.Event()
        self._error = None
        self._start_time = time.monotonic()
        self._opened = True
        self._closed = False

        # 线程非 daemon（§4.3）：close() 内 join 等待其退出，避免遗留后台线程
        self._thread = threading.Thread(target=self._produce, name=THREAD_NAME, daemon=False)
        self._thread.start()

    def read(self) -> Optional[Frame]:
        """阻塞取一帧；线程已退出且队列耗尽时返回 None（读尽）。

        队列中已捕获的帧先交付，全部取空后才暴露流错误：
        断流错误在「线程已退出 + 队列已空」时才抛 InputSourceError。
        """
        if not self._opened:
            raise InputSourceError("read() 之前必须先调用 open()")

        while True:
            try:
                return self._queue.get(timeout=self._cfg.timeout)
            except queue.Empty:
                pass

            # 队列暂时为空 ≠ 读尽：先看线程是否已退出，未退出则继续等待
            thread_dead = self._thread is not None and not self._thread.is_alive()
            if thread_dead and self._queue.empty():
                if self._error is not None:
                    raise self._error
                return None

    def close(self) -> None:
        """停止后台线程并释放句柄；幂等。"""
        if self._closed:
            return
        self._closed = True

        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(self._cfg.timeout)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        # 清空缓冲，使 close 之后的 read() 直接返回 None（读尽语义保持一致）
        if self._queue is not None:
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break

    # ---------- 生产者（后台线程） ----------

    def _produce(self) -> None:
        """读帧入队；断流按 §4.4 策略重连，重连用尽置错误并退出。"""
        cfg = self._cfg
        index = 0
        failures = 0

        while not self._stop.is_set():
            ok, image = self._cap.read()
            if ok and image is not None and image.size:
                height, width = image.shape[:2]
                if height == 0 or width == 0:
                    self._error = InputSourceError(f"摄像头帧尺寸为 0（源帧号 {index}）")
                    return
                failures = 0
                if index % cfg.step == 0:  # 跳帧：只把每 step 帧入队，帧号照常推进
                    self._enqueue(image, width, height, index)
                index += 1
                continue

            if self._stop.is_set():
                return
            if cfg.reconnect and failures < cfg.reconnect_max:
                failures += 1
                if self._stop.wait(cfg.reconnect_interval):  # 可被 close 打断的等待
                    return
                self._reopen()
                continue

            self._error = InputSourceError(
                f"摄像头流中断，重连 {failures} 次后仍未恢复：{self._describe(cfg)}"
            )
            return

    def _enqueue(self, image: np.ndarray, width: int, height: int, index: int) -> None:
        """入队；队列满则丢弃最旧帧，保留最新帧（§4.3 队列满策略）。"""
        frame = Frame(
            image=image,
            width=width,
            height=height,
            frame_index=index,
            timestamp=time.monotonic() - self._start_time,  # 秒，从源起点算
        )
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            try:
                self._queue.get_nowait()  # 丢旧
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(frame)  # 补新
            except queue.Full:
                pass

    def _reopen(self) -> None:
        """重连：释放旧句柄并按原目标重建（close 期间可能被并发置空）。"""
        if self._stop.is_set() or self._cap is None:
            return
        self._cap.release()
        self._cap = self._build_capture(self._cfg)

    # ---------- 内部 ----------

    @staticmethod
    def _build_capture(cfg: SourceConfig) -> cv2.VideoCapture:
        """按 device_id / url 建句柄；RTSP 将内部缓冲压到 1 以降低延迟。"""
        if cfg.device_id is not None:
            return cv2.VideoCapture(cfg.device_id)
        cap = cv2.VideoCapture(cfg.url)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    @staticmethod
    def _describe(cfg: SourceConfig) -> str:
        return f"device_id={cfg.device_id}" if cfg.device_id is not None else f"url={cfg.url}"
