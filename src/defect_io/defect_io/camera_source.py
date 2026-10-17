"""摄像头输入源：本地 USB / RTSP 网络流（实现文档 阶段2 · 4.1、4.3）。

后台线程读帧入队，避免检测慢于帧率时积压；主流程只取最新帧。
"""

import threading
import time
from queue import Empty, Queue
from typing import Optional

import cv2

from defect_core.types import Frame

from .input_source import InputSource


class CameraSource(InputSource):
    """摄像头 / RTSP 输入源，生产者-消费者模式保证实时性。"""

    def __init__(self) -> None:
        self._cap: Optional[cv2.VideoCapture] = None
        self._queue: Queue = Queue(maxsize=2)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._source = ''
        self._index = 0

    def open(self, source_cfg: dict) -> None:
        source = source_cfg.get('source', 0)
        # 纯数字字符串按设备号处理，其余按 URL 处理（RTSP）
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f'无法打开摄像头/视频流: {source}')

        self._source = str(source)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        """后台读帧线程：队列满时丢弃旧帧，始终保留最新。"""
        while self._running and self._cap is not None:
            ok, image = self._cap.read()
            if not ok:
                # TODO(阶段8): 断流检测与自动重连
                time.sleep(0.05)
                continue
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except Empty:
                    pass
            self._queue.put(image)

    def read(self) -> Optional[Frame]:
        try:
            image = self._queue.get(timeout=1.0)
        except Empty:
            return None

        height, width = image.shape[:2]
        self._index += 1
        return Frame(
            image=image,
            width=width,
            height=height,
            frame_index=self._index,
            source_id=self._source,
        )

    def close(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
