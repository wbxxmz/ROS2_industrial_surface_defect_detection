"""输入源统一抽象接口（实现文档 阶段0 · 5.3 节，签名已冻结）。"""

from abc import ABC, abstractmethod
from typing import Optional

from defect_core.types import Frame


class InputSource(ABC):
    """图片 / 视频 / 摄像头统一基类，read() 统一产出 Frame。"""

    @abstractmethod
    def open(self, source_cfg: dict) -> None:
        """打开输入源；失败时抛出可读异常。"""

    @abstractmethod
    def read(self) -> Optional[Frame]:
        """读取一帧；无更多数据时返回 None。"""

    @abstractmethod
    def close(self) -> None:
        """关闭输入源，释放句柄。"""
