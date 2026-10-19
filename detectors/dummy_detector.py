"""DummyDetector：冒烟测试用最小检测器（空实现，不做任何检测）。"""

from typing import List

from core import Detection, Detector, Frame


class DummyDetector(Detector):
    """最小实现：name/version 固定，detect 恒返回空列表。"""

    @property
    def name(self) -> str:
        return "dummy"

    @property
    def version(self) -> str:
        return "0.1.0"

    def load(self, cfg: dict) -> None:
        """空实现。"""
        return None

    def detect(self, frame: Frame) -> List[Detection]:
        """空实现：恒返回空列表（无缺陷）。"""
        return []

    def release(self) -> None:
        """空实现。"""
        return None
