"""core 层数据结构：帧对象与检测结果。

字段与阶段 0 §5.4.1 / §5.4.4 契约一致；字段冻结，只允许新增可选字段。
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Frame:
    """统一帧对象（阶段 0 §5.4.4）。"""

    image: np.ndarray  # uint8, H×W×3, BGR；需 RGB 的算法在检测器内部自行转换
    width: int  # 像素，对应 image.shape[1]
    height: int  # 像素，对应 image.shape[0]
    frame_index: int  # 从 0 起的源帧序号；跳帧时仍为源帧号
    timestamp: float  # 秒，从源起点算；图片为 0.0


@dataclass
class Detection:
    """统一检测结果（阶段 0 §5.4.1）。"""

    class_id: int
    class_name: str
    bbox: tuple[int, int, int, int]  # 像素，(x, y, w, h)，原点左上，不使用归一化坐标
    area_px: int  # 缺陷实际像素面积，≠ bbox 矩形面积
    area_ratio: float  # 0~1，缺陷面积 / 图像面积
    confidence: float  # 0.0–1.0
    mask: Optional[np.ndarray] = None  # 像素级；数据类型 TBD-阶段 3
    severity: Optional[int] = None  # 阶段 6 §4.1.1：0=轻度 / 1=中度 / 2=重度；None=未分级
