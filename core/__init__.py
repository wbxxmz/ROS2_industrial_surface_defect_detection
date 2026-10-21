"""core：接口与数据结构（Detection / Detector / InputSource / Frame）。"""

from core.detector import Detector
from core.input_source import InputSource
from core.types import Detection, Frame
from core.verdict import Verdict

__all__ = ["Detection", "Frame", "Verdict", "Detector", "InputSource"]
