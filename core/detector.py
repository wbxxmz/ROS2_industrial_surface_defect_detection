"""Detector 抽象基类（阶段 0 §5.4.2，签名冻结）。

基类方法体只抛 NotImplementedError，表示"未实现"；
子类覆盖后按阶段 0 契约抛 DetectorLoadError / DetectorInferenceError，
不得把 NotImplementedError 当作契约异常。
"""

from abc import ABC, abstractmethod
from typing import List

from core.types import Detection, Frame


class Detector(ABC):
    """检测器抽象基类。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """算法标识：str，不可 None。"""
        raise NotImplementedError

    @property
    @abstractmethod
    def version(self) -> str:
        """算法版本：str；格式 TBD-阶段 1；不可 None。"""
        raise NotImplementedError

    @abstractmethod
    def load(self, cfg: dict) -> None:
        """加载/初始化算法。

        :param cfg: 配置 dict；字段 TBD-阶段 2
        :return: None
        :raises DetectorLoadError: 加载失败时抛出，由调用方捕获
        """
        raise NotImplementedError

    @abstractmethod
    def detect(self, frame: Frame) -> List[Detection]:
        """检测一帧。

        :param frame: 统一帧对象
        :return: List[Detection]；无缺陷返回空列表 []，不返回 None
        :raises ValueError: 输入非法
        :raises DetectorInferenceError: 内部推理异常
        """
        raise NotImplementedError

    @abstractmethod
    def release(self) -> None:
        """释放资源；返回 None；应幂等（细节 TBD-阶段 2）。"""
        raise NotImplementedError
