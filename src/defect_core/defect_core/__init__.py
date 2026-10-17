"""defect_core：统一检测接口、算法注册与检测流水线。"""

from .types import (CLASS_NAMES, CLASS_DENT, CLASS_NOTCH, CLASS_SCRATCH,
                    CLASS_STAIN, Detection, Frame, InspectionResult)
from .detector_base import Detector
from .image_utils import imgmsg_to_numpy, numpy_to_imgmsg
from .registry import available_detectors, create_detector, register_detector

__all__ = [
    'imgmsg_to_numpy', 'numpy_to_imgmsg',
    'Detection', 'Frame', 'InspectionResult',
    'CLASS_SCRATCH', 'CLASS_DENT', 'CLASS_STAIN', 'CLASS_NOTCH', 'CLASS_NAMES',
    'Detector',
    'register_detector', 'create_detector', 'available_detectors',
]
