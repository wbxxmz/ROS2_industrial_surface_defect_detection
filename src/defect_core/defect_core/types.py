# 统一检测结果数据结构（对应实现文档 阶段0 · 5.1 节，字段已冻结）

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# 缺陷类别常量，与 defect_interfaces/msg/Defect.msg 保持一致
CLASS_UNKNOWN = 0
CLASS_SCRATCH = 1
CLASS_DENT = 2
CLASS_STAIN = 3
CLASS_NOTCH = 4

CLASS_NAMES = {
    CLASS_UNKNOWN: 'unknown',
    CLASS_SCRATCH: 'scratch',
    CLASS_DENT: 'dent',
    CLASS_STAIN: 'stain',
    CLASS_NOTCH: 'notch',
}


@dataclass
class Detection:
    """单个缺陷检测结果，所有算法（OpenCV / YOLO）统一输出该结构。"""

    class_id: int
    class_name: str
    bbox: tuple            # (x, y, w, h)，图像像素坐标，左上角为原点
    confidence: float      # 置信度 0.0~1.0
    area_px: int = 0       # 像素面积
    area_ratio: float = 0.0  # 相对图像面积占比 0.0~1.0
    mask: Optional[np.ndarray] = None  # 可选分割掩码
    severity: int = 0      # 严重程度等级，阶段6 启用（阶段0 预留）


@dataclass
class Frame:
    """统一帧对象（对应实现文档 阶段2 · 4.2 节）。"""

    image: np.ndarray
    width: int
    height: int
    frame_index: int = 0
    timestamp: float = 0.0
    source_id: str = ''


@dataclass
class InspectionResult:
    """单帧检测结果汇总（对应实现文档 阶段5）。"""

    source_id: str
    detector_name: str = ''
    detector_version: str = ''
    is_pass: bool = True
    defects: list = field(default_factory=list)

    @property
    def total_defects(self) -> int:
        return len(self.defects)
