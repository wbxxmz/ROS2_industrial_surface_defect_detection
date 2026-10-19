"""InputSource 抽象基类（阶段 0 §5.4.3，签名冻结）。

基类方法体只抛 NotImplementedError，表示"未实现"；
子类覆盖后按阶段 0 契约抛 SourceOpenError / InputSourceError，
不得把 NotImplementedError 当作契约异常。
"""

from abc import ABC, abstractmethod
from typing import Optional

from core.types import Frame


class InputSource(ABC):
    """输入源抽象基类。"""

    @abstractmethod
    def open(self, source_cfg: dict) -> None:
        """打开来源。

        :param source_cfg: 配置 dict；字段 TBD-阶段 2
        :return: None
        :raises SourceOpenError: 打开失败时抛出，由调用方捕获
        """
        raise NotImplementedError

    @abstractmethod
    def read(self) -> Optional[Frame]:
        """读一帧，统一返回帧对象。

        :return: Frame 或 None；None 只表示读尽/结束，不表示出错
        :raises InputSourceError: 运行中断流/出错
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """关闭来源；返回 None；应幂等（细节 TBD-阶段 2）。"""
        raise NotImplementedError
