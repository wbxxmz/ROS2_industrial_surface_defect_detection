"""流水线骨架：采集 → 预处理(占位) → 检测 → 后处理(占位) → 输出。

仅直通，不含任何算法处理；阶段 2–5 逐个"填空"。
"""

import logging
from typing import List

import numpy as np

from core import Detector, Detection, Frame


def build_detector(config: dict) -> Detector:
    """按 config["detector"]["type"] 选择检测器（本阶段仅 dummy）。

    :raises ValueError: 未知 detector.type
    """
    detector_type = (config.get("detector") or {}).get("type")
    if detector_type == "dummy":
        from detectors.dummy_detector import DummyDetector

        return DummyDetector()
    raise ValueError(f"未知 detector.type: {detector_type!r}")


class Pipeline:
    """检测流水线骨架。"""

    def __init__(self, config: dict, detector: Detector) -> None:
        self._config = config
        self._detector = detector
        self._logger = logging.getLogger("defect.pipeline")

    def capture(self) -> Frame:
        """采集一帧（本阶段占位；阶段 2 按 config["input"]["type"] 构造 InputSource）。"""
        image = np.zeros((1, 1, 3), dtype=np.uint8)
        return Frame(image=image, width=1, height=1, frame_index=0, timestamp=0.0)

    def preprocess(self, frame: Frame) -> Frame:
        """预处理（直通）：原样返回 Frame。"""
        return frame

    def detect(self, frame: Frame) -> List[Detection]:
        """检测：委托给 Detector。"""
        return self._detector.detect(frame)

    def postprocess(self, detections: List[Detection]) -> List[Detection]:
        """后处理（直通）：原样返回 List[Detection]。"""
        return detections

    def run(self) -> List[Detection]:
        """跑完整条流水线，返回结果列表。"""
        self._detector.load(self._config.get("detector") or {})
        try:
            frame = self.preprocess(self.capture())
            detections = self.postprocess(self.detect(frame))
        finally:
            self._detector.release()
        self._logger.info(
            "流水线完成：帧 %s×%s，检测结果 %d 条", frame.width, frame.height, len(detections)
        )
        return detections
