"""传统 OpenCV 检测器（实现文档 阶段3）。

当前为骨架：load/detect/release 已按统一接口落位，
detect 返回空列表，待阶段3 填入 预处理 → 特征提取 → 后处理 流程。
"""

from typing import List

from ..detector_base import Detector
from ..registry import register_detector
from ..types import Detection, Frame


@register_detector('opencv')
class OpencvDetector(Detector):
    """基于传统图像处理的缺陷检测器，无需训练数据。"""

    def __init__(self) -> None:
        self._cfg: dict = {}
        self._loaded = False

    @property
    def name(self) -> str:
        return 'opencv'

    @property
    def version(self) -> str:
        return '0.1.0'

    def load(self, cfg: dict) -> None:
        # TODO(阶段3): 读取阈值、形态学核、面积上下限等参数
        self._cfg = dict(cfg or {})
        self._loaded = True

    def detect(self, frame: Frame) -> List[Detection]:
        # TODO(阶段3): 预处理 → 候选区域提取 → 几何/面积过滤 → 分类 → Detection 组装
        return []

    def release(self) -> None:
        self._loaded = False
