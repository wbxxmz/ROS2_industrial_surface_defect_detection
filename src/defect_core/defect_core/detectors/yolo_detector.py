"""YOLO 系列检测器（实现文档 阶段4）。

当前为骨架：torch/ultralytics 为可选依赖，未安装时 is_available() 返回 False，
load() 抛出可读异常，不阻塞 opencv 路线的使用。
"""

from typing import List

from ..detector_base import Detector
from ..registry import register_detector
from ..types import Detection, Frame


def is_available() -> bool:
    """依赖探测：torch 与 ultralytics 是否可用。"""
    try:
        import torch  # noqa: F401
        import ultralytics  # noqa: F401
    except ImportError:
        return False
    return True


@register_detector('yolo')
class YoloDetector(Detector):
    """YOLO 系列深度学习检测器，需标注数据训练 / 预训练权重推理。"""

    def __init__(self) -> None:
        self._model = None
        self._cfg: dict = {}

    @property
    def name(self) -> str:
        return 'yolo'

    @property
    def version(self) -> str:
        # TODO(阶段4): 返回权重版本，用于报告溯源
        return '0.1.0'

    def load(self, cfg: dict) -> None:
        # TODO(阶段4): 加载权重、设置 conf/iou 阈值与设备
        if not is_available():
            raise RuntimeError(
                '未安装 torch/ultralytics，无法使用 YOLO 检测器；'
                '请先安装依赖或改用 opencv 检测器')
        self._cfg = dict(cfg or {})

    def detect(self, frame: Frame) -> List[Detection]:
        # TODO(阶段4): 推理 → 置信度阈值过滤 → NMS → 映射为 Detection
        return []

    def release(self) -> None:
        # TODO(阶段4): 释放模型与显存
        self._model = None
