# 检测器统一抽象接口（对应实现文档 阶段0 · 5.2 节，签名已冻结，不可破坏）

from abc import ABC, abstractmethod
from typing import List

from .types import Detection, Frame


class Detector(ABC):
    """所有检测算法的基类：新增算法只需继承本类并注册。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """算法标识，如 opencv / yolov8n。用于界面展示与报告溯源。"""

    @property
    def version(self) -> str:
        """算法/权重版本，用于报告溯源。"""
        return '0.0.0'

    def load(self, cfg: dict) -> None:
        """加载与初始化算法，cfg 为算法参数字典。"""
        raise NotImplementedError

    @abstractmethod
    def detect(self, frame: Frame) -> List[Detection]:
        """核心方法：对单帧执行检测并返回统一结构的缺陷列表。"""

    def release(self) -> None:
        """释放资源（GPU 显存、模型句柄等）。"""
        return None
