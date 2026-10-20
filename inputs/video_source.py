"""阶段 2 §4.1 / §4.3：VideoSource（逐帧读取、跳帧、区间、循环）。

契约（阶段 0 §5.4.3）：
- ``open(source_cfg)``：失败抛 SourceOpenError（文件不存在 / 无法打开）。
- ``read()``：返回 Frame；读到末尾返回 None；中途解码失败抛 InputSourceError。
- ``close()``：返回 None，幂等。
跳帧语义（§4.3）：step=N 时每 N 帧产出一次，frame_index 仍为源帧序号（0, 3, 6 …）。
"""

from pathlib import Path
from typing import Optional

import cv2

from core.input_source import InputSource
from core.types import Frame
from inputs.config import SourceConfig
from utils.exceptions import InputSourceError, SourceOpenError

DEFAULT_FPS = 25.0


class VideoSource(InputSource):
    """本地视频文件输入源。"""

    def __init__(self) -> None:
        self._cfg: Optional[SourceConfig] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._start = 0
        self._next_index = 0
        self._fps = DEFAULT_FPS
        self._rewind_failed = False
        self._opened = False
        self._closed = False

    def open(self, source_cfg: dict) -> None:
        """打开视频文件；source_cfg 字段见阶段 2 §4.4。"""
        cfg = SourceConfig.from_dict(source_cfg)
        if cfg.type != "video":
            raise ValueError(f"VideoSource 只接受 type='video'，收到 {cfg.type!r}")

        if not Path(cfg.path).is_file():
            raise SourceOpenError(f"视频文件不存在或不是文件：{cfg.path}")

        cap = cv2.VideoCapture(cfg.path)
        if not cap.isOpened():
            cap.release()
            raise SourceOpenError(f"视频无法打开（编码不支持或文件损坏）：{cfg.path}")

        self._cfg = cfg
        self._cap = cap
        fps = cap.get(cv2.CAP_PROP_FPS)
        self._fps = fps if 0 < fps <= 1000 else DEFAULT_FPS
        self._start = cfg.start_frame or 0
        self._next_index = self._start
        self._rewind_failed = False
        if self._start > 0:
            self._seek(self._start)
        self._opened = True
        self._closed = False

    def read(self) -> Optional[Frame]:
        """读一帧；读到末尾（或区间末尾）返回 None。"""
        if not self._opened:
            raise InputSourceError("read() 之前必须先调用 open()")
        if self._closed:
            return None

        cfg = self._cfg
        while True:
            index = self._next_index
            if cfg.end_frame is not None and index > cfg.end_frame:
                if cfg.loop:
                    self._rewind()
                    continue
                return None

            image = self._grab()
            if image is None:
                if cfg.loop:
                    self._rewind()
                    continue
                return None

            self._next_index = index + 1
            self._rewind_failed = False
            if (index - self._start) % cfg.step:
                continue  # 跳帧：不产出，但源帧号照常推进

            height, width = image.shape[:2]
            if height == 0 or width == 0:
                raise InputSourceError(f"视频帧尺寸为 0（源帧号 {index}）")
            return Frame(
                image=image,
                width=width,
                height=height,
                frame_index=index,
                timestamp=index / self._fps,  # 秒，从源起点算（§5.4.4）
            )

    def close(self) -> None:
        """关闭视频句柄；幂等（重复调用不重复释放）。"""
        if self._closed:
            return
        self._closed = True
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ---------- 内部 ----------

    def _grab(self) -> Optional["cv2.typing.MatLike"]:
        """取一帧原始图像；返回 None 表示读尽，中途失败抛 InputSourceError。"""
        ok, frame = self._cap.read()
        if ok and frame is not None and frame.size:
            return frame
        # 区分「读尽」与「中途解码失败」：位置未到末尾却读失败即视为流中断
        total = self._cap.get(cv2.CAP_PROP_FRAME_COUNT)
        pos = self._cap.get(cv2.CAP_PROP_POS_FRAMES)
        if total > 0 and pos < total - 1:
            raise InputSourceError(f"视频读取中途失败：位置 {int(pos)} / 总帧数 {int(total)}")
        return None

    def _seek(self, frame_index: int) -> None:
        """定位到指定源帧号。"""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        self._next_index = frame_index

    def _rewind(self) -> None:
        """循环播放：回到区间起点；定位失败则显式报错，避免死循环。"""
        if self._rewind_failed:
            raise InputSourceError("视频循环重定位失败，无法回到起点")
        self._rewind_failed = True
        self._seek(self._start)
