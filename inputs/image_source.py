"""阶段 2 §4.1：ImageSource（单张图片 + 目录批量导入）。

契约（阶段 0 §5.4.3）：
- ``open(source_cfg)``：失败抛 SourceOpenError（文件不存在 / 无法解码 / 目录为空）。
- ``read()``：返回 Frame；真正读完返回 None；解码失败抛 InputSourceError。
- ``close()``：返回 None，幂等。
帧对象字段见阶段 0 §5.4.4：图片的 timestamp 恒为 0.0，frame_index 为目录内序号。
"""

from pathlib import Path
from typing import List, Optional

import cv2

from core.input_source import InputSource
from core.types import Frame
from inputs.config import SourceConfig
from utils.exceptions import InputSourceError, SourceOpenError


class ImageSource(InputSource):
    """本地图片 / 目录输入源。"""

    def __init__(self) -> None:
        self._cfg: Optional[SourceConfig] = None
        self._paths: List[Path] = []
        self._cursor = 0
        self._opened = False
        self._closed = False

    def open(self, source_cfg: dict) -> None:
        """打开来源；source_cfg 字段见阶段 2 §4.4。"""
        cfg = SourceConfig.from_dict(source_cfg)
        if cfg.type != "image":
            raise ValueError(f"ImageSource 只接受 type='image'，收到 {cfg.type!r}")

        if cfg.path is not None:
            paths = [self._validated_single(Path(cfg.path))]
        else:
            paths = self._collect_dir(Path(cfg.dir), cfg.extensions)

        self._cfg = cfg
        self._paths = paths
        self._cursor = 0
        self._opened = True
        self._closed = False

    def read(self) -> Optional[Frame]:
        """读一张图片；读完返回 None。"""
        if not self._opened:
            raise InputSourceError("read() 之前必须先调用 open()")
        if self._closed or self._cursor >= len(self._paths):
            return None

        path = self._paths[self._cursor]
        # 按需读取，不整体载入（§8 大批量内存约束）
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            # 解码失败不跳过、不推进游标：错误必须显式暴露（§5 read 边界）
            raise InputSourceError(f"图片解码失败或为空：{path}")

        height, width = image.shape[:2]
        if height == 0 or width == 0:
            raise InputSourceError(f"图片尺寸为 0：{path}")

        frame_index = self._cursor
        self._cursor += 1
        return Frame(
            image=image,
            width=width,
            height=height,
            frame_index=frame_index,
            timestamp=0.0,  # 阶段 0 §5.4.4：图片为 0.0
        )

    @property
    def current_path(self) -> Optional[Path]:
        """最近一次 ``read()`` 返回的图片路径；未打开或已读尽时为 None。

        阶段 7 需要"真实文件名"作为样本名：``Frame`` 契约为阶段 0 冻结字段，
        不新增字段，改由输入源暴露当前路径。
        """
        if not self._opened or self._cursor == 0:
            return None
        return self._paths[self._cursor - 1]

    def close(self) -> None:
        """关闭来源；幂等（无外部句柄，仅置标志）。"""
        if self._closed:
            return
        self._closed = True

    # ---------- 内部 ----------

    @staticmethod
    def _validated_single(path: Path) -> Path:
        """单张图片：open 阶段即校验存在性与可解码性（§5 open 边界）。"""
        if not path.is_file():
            raise SourceOpenError(f"图片不存在或不是文件：{path}")
        probe = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if probe is None or probe.size == 0:
            raise SourceOpenError(f"图片无法解码或为空：{path}")
        return path

    @staticmethod
    def _collect_dir(directory: Path, extensions: List[str]) -> List[Path]:
        """目录批量：按文件名排序，扩展名白名单过滤（§7 按序、无重复无遗漏）。"""
        if not directory.is_dir():
            raise SourceOpenError(f"目录不存在或不是目录：{directory}")
        allowed = {ext.lower() for ext in extensions}
        paths = sorted(
            p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in allowed
        )
        if not paths:
            raise SourceOpenError(f"目录内未找到匹配 {sorted(allowed)} 的图片：{directory}")
        return paths
