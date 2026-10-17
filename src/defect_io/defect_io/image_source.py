"""图片输入源：单张读取 + 批量目录导入（实现文档 阶段2 · 4.1）。"""

from pathlib import Path
from typing import List, Optional

import cv2

from defect_core.types import Frame

from .input_source import InputSource

_SUPPORTED_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


class ImageSource(InputSource):
    """支持单张图片与目录批量导入，按文件名排序逐张产出。"""

    def __init__(self) -> None:
        self._files: List[Path] = []
        self._cursor = 0
        self._index = 0

    def open(self, source_cfg: dict) -> None:
        path = Path(source_cfg.get('path', ''))
        if not path.exists():
            raise FileNotFoundError(f'图片路径不存在: {path}')

        if path.is_dir():
            # 批量导入：只收支持的图片格式并按名称排序
            self._files = sorted(
                p for p in path.iterdir() if p.suffix.lower() in _SUPPORTED_SUFFIXES)
            if not self._files:
                raise FileNotFoundError(f'目录中未找到图片文件: {path}')
        else:
            self._files = [path]
        self._cursor = 0

    def read(self) -> Optional[Frame]:
        while self._cursor < len(self._files):
            current = self._files[self._cursor]
            self._cursor += 1
            image = cv2.imread(str(current))
            if image is None:
                # 损坏文件跳过并记录，不中断批量任务（阶段8 要求）
                continue
            height, width = image.shape[:2]
            self._index += 1
            return Frame(
                image=image,
                width=width,
                height=height,
                frame_index=self._index,
                source_id=str(current),
            )
        return None

    def close(self) -> None:
        self._files = []
        self._cursor = 0
