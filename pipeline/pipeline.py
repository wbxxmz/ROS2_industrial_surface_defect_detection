"""流水线骨架：采集 → 预处理(占位) → 检测 → 后处理(占位) → 输出。

阶段 2 起「采集」由 ``InputSource`` 承接：``capture()`` 委托 ``InputSource.read()``，
``run()`` 逐帧消费直到读尽（``None``），并负责 ``open()`` / ``close()`` 生命周期。
预处理、后处理仍为直通占位，阶段 3–5 逐个"填空"。
"""

import logging
from typing import List, Optional

from core import Detector, Detection, Frame, InputSource


def build_detector(config: dict) -> Detector:
    """按 config["detector"]["type"] 选择检测器（dummy / opencv / yolo）。

    参数校验由各检测器的 ``load`` 在运行时完成（非法参数抛 DetectorLoadError）。

    :raises ValueError: 未知 detector.type
    """
    detector_type = (config.get("detector") or {}).get("type")
    if detector_type == "dummy":
        from detectors.dummy_detector import DummyDetector

        return DummyDetector()
    if detector_type == "opencv":
        from detectors.opencv_detector import OpencvDetector

        return OpencvDetector()
    if detector_type == "yolo":
        from detectors.yolo_detector import YoloDetector

        return YoloDetector()
    raise ValueError(f"未知 detector.type: {detector_type!r}")


class Pipeline:
    """检测流水线骨架。"""

    def __init__(self, config: dict, detector: Detector, source: InputSource) -> None:
        self._config = config
        self._detector = detector
        self._source = source
        self._logger = logging.getLogger("defect.pipeline")

    def capture(self) -> Optional[Frame]:
        """采集一帧：委托给 InputSource.read()；读尽返回 None。"""
        return self._source.read()

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
        """跑完整条流水线：逐帧采集直到读尽，返回累计结果列表。"""
        results: List[Detection] = []
        try:
            self._detector.load(self._config.get("detector") or {})
            self._source.open(self._config.get("input") or {})
            while True:
                frame = self.capture()
                if frame is None:  # None 只表示读尽（阶段 0 §5.4.3）
                    break
                detections = self.postprocess(self.detect(self.preprocess(frame)))
                results.extend(detections)
                self._logger.info(
                    "帧 %d：%s×%s，检测结果 %d 条",
                    frame.frame_index,
                    frame.width,
                    frame.height,
                    len(detections),
                )
        finally:
            self._detector.release()
            self._source.close()
        self._logger.info("流水线完成：累计 %d 条检测结果", len(results))
        return results
