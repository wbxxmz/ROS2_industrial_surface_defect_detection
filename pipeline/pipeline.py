"""流水线骨架：采集 → 预处理(占位) → 检测 → 后处理(占位) → 输出。

阶段 2 起「采集」由 ``InputSource`` 承接：``capture()`` 委托 ``InputSource.read()``，
``run()`` 逐帧消费直到读尽（``None``），并负责 ``open()`` / ``close()`` 生命周期。
预处理、后处理仍为直通占位，阶段 3–5 逐个"填空"。
"""

import logging
from pathlib import Path
from typing import List, Optional

from core import Detector, Detection, Frame, InputSource
from judge import judge
from severity import grade
from visualization import render_to_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VISUAL_OUTPUT_DIR = PROJECT_ROOT / "reports" / "visual"  # 已在 .gitignore 中屏蔽


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
        # 阶段 7：逐样本记录（source / verdict / detections / image_path），供报告模块消费
        self.samples: List[dict] = []

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

    def grade(self, detections: List[Detection]) -> List[Detection]:
        """阶段 6 §4.1.7：为每条检测写入 severity（detect 与 judge 之间）。"""
        return grade(detections, self._config.get("severity"))

    def _source_label(self) -> str:
        """输出文件名前缀：图片/视频取文件名主干，摄像头取类型名。"""
        input_cfg = self._config.get("input") or {}
        for key in ("path", "dir"):
            value = input_cfg.get(key)
            if value:
                return Path(str(value)).stem or "frame"
        return str(input_cfg.get("type") or "frame")

    def _sample_name(self, frame: Frame) -> str:
        """样本名：图片源取真实文件名主干（如 ``crazing_98``），其余源回退为「源标识_frameN」。

        图片目录批量导入时每张图都是独立的 frame_index，若统一拼 ``源标识_frameN``
        会出现多张图难以区分的情况，故优先使用输入源暴露的当前文件路径。
        """
        current = getattr(self._source, "current_path", None)
        if current is not None:
            return Path(str(current)).stem
        return f"{self._source_label()}_frame{frame.frame_index:06d}"

    def _visual_path(self, sample_name: str) -> Path:
        """标注图落盘路径：reports/visual/{样本名}.jpg（§4.4）。"""
        return VISUAL_OUTPUT_DIR / f"{sample_name}.jpg"

    def run(self) -> List[Detection]:
        """跑完整条流水线：逐帧采集直到读尽，判定并落盘标注图，返回累计结果列表。"""
        results: List[Detection] = []
        judge_cfg = self._config.get("judge")
        try:
            self._detector.load(self._config.get("detector") or {})
            self._source.open(self._config.get("input") or {})
            while True:
                frame = self.capture()
                if frame is None:  # None 只表示读尽（阶段 0 §5.4.3）
                    break
                detections = self.grade(self.postprocess(self.detect(self.preprocess(frame))))
                results.extend(detections)
                verdict = judge(detections, judge_cfg)
                sample_name = self._sample_name(frame)
                output_path = render_to_file(frame, detections, verdict, str(self._visual_path(sample_name)))
                self.samples.append(
                    {
                        "source": sample_name,
                        "verdict": verdict,
                        "detections": detections,
                        "image_path": str(output_path),
                    }
                )
                self._logger.info(
                    "帧 %d：%s×%s，检测结果 %d 条，is_pass=%s，max_severity=%s，reason=%s，标注图=%s",
                    frame.frame_index,
                    frame.width,
                    frame.height,
                    len(detections),
                    verdict.is_pass,
                    verdict.max_severity,
                    verdict.reason,
                    output_path,
                )
        finally:
            self._detector.release()
            self._source.close()
        self._logger.info("流水线完成：累计 %d 条检测结果", len(results))
        return results
