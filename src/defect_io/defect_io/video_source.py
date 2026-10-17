"""视频文件输入源：逐帧读取、抽帧/跳帧（实现文档 阶段2 · 4.1）。"""

from pathlib import Path
from typing import Optional

import cv2

from defect_core.types import Frame

from .input_source import InputSource


class VideoSource(InputSource):
    """本地视频文件输入源，支持跳帧以兼顾实时性与算力。"""

    def __init__(self) -> None:
        self._cap: Optional[cv2.VideoCapture] = None
        self._path = ''
        self._stride = 1
        self._index = 0

    def open(self, source_cfg: dict) -> None:
        path = Path(source_cfg.get('path', ''))
        if not path.exists():
            raise FileNotFoundError(f'视频文件不存在: {path}')

        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise RuntimeError(f'无法打开视频文件: {path}')

        self._path = str(path)
        self._stride = max(1, int(source_cfg.get('frame_stride', 1)))
        self._index = 0

    def read(self) -> Optional[Frame]:
        if self._cap is None:
            return None

        # 跳帧：仅在检测帧上做全流程处理
        for _ in range(self._stride):
            ok = self._cap.grab()
            if not ok:
                return None

        ok, image = self._cap.retrieve()
        if not ok or image is None:
            return None

        height, width = image.shape[:2]
        self._index += 1
        return Frame(
            image=image,
            width=width,
            height=height,
            frame_index=self._index,
            timestamp=self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0,
            source_id=f'{self._path}#{self._index}',
        )

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
