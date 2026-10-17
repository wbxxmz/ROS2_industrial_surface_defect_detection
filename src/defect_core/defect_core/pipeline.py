"""检测流水线编排（实现文档 阶段1 骨架）。

流程：帧 → 预处理 → 检测 → 后处理 → InspectionResult
预处理 / 后处理当前为直通（pass-through）占位，阶段3 起逐个填空。
"""

from typing import Callable, List, Optional

from .detector_base import Detector
from .types import Detection, Frame, InspectionResult


class Pipeline:
    """串接"采集 → 预处理 → 检测 → 后处理 → 判定"的可替换流水线。"""

    def __init__(self, detector: Detector, judge: Optional[Callable] = None) -> None:
        self._detector = detector
        self._judge = judge
        self._preprocess: List[Callable] = []
        self._postprocess: List[Callable] = []

    def set_detector(self, detector: Detector) -> None:
        """运行时替换检测算法（对应 /set_detector 服务）。"""
        self._detector = detector

    def add_preprocess(self, fn: Callable) -> None:
        self._preprocess.append(fn)

    def add_postprocess(self, fn: Callable) -> None:
        self._postprocess.append(fn)

    def run(self, frame: Frame) -> InspectionResult:
        image = frame.image
        # 预处理（占位：直通）
        for fn in self._preprocess:
            image = fn(image)
        frame.image = image

        # 检测
        defects: List[Detection] = self._detector.detect(frame)

        # 后处理（占位：直通）
        for fn in self._postprocess:
            defects = fn(defects, frame)

        result = InspectionResult(
            source_id=frame.source_id,
            detector_name=self._detector.name,
            detector_version=self._detector.version,
            defects=defects,
        )
        # 判定（阶段5）
        if self._judge is not None:
            result.is_pass = self._judge(result)
        return result
