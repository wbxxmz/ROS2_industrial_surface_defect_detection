"""core：接口与数据结构（Detection / Detector / InputSource / Frame）。"""

from core.detector import Detector
from core.input_source import InputSource
from core.types import Detection, Frame

__all__ = ["Detection", "Frame", "Detector", "InputSource"]
