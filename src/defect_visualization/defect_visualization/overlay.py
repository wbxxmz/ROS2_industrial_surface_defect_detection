"""叠绘模块（实现文档 阶段5 · 4.1）。

不同缺陷类别使用固定颜色映射，便于一眼区分；预留严重度展示位。
"""

from typing import List, Optional

import cv2

from defect_core.types import (CLASS_DENT, CLASS_NOTCH, CLASS_SCRATCH,
                               CLASS_STAIN, Detection)

# 类别 → 颜色（BGR），全平台统一，供报告与界面复用
CLASS_COLORS = {
    CLASS_SCRATCH: (0, 0, 255),    # 划痕 红
    CLASS_DENT: (255, 0, 0),       # 凹坑 蓝
    CLASS_STAIN: (0, 255, 0),      # 污渍 绿
    CLASS_NOTCH: (0, 165, 255),    # 缺口 橙
}
_DEFAULT_COLOR = (255, 255, 255)


def draw_detections(image, detections: List[Detection], show_confidence: bool = True):
    """在图像上叠加边界框、类别标签、置信度与面积标注。"""
    canvas = image.copy()
    height, width = canvas.shape[:2]
    # 按分辨率缩放线宽与字号，避免高分辨率下标注过小
    scale = max(1.0, min(width, height) / 640.0)
    thickness = max(1, int(round(scale)))
    font_scale = 0.5 * scale

    for det in detections:
        x, y, w, h = (int(v) for v in det.bbox)
        color = CLASS_COLORS.get(det.class_id, _DEFAULT_COLOR)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness)

        if show_confidence:
            label = f'{det.class_name} {det.confidence:.2f}'
            if det.area_px:
                label += f' area={det.area_px}px'
            cv2.putText(canvas, label, (x, max(y - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

        if det.mask is not None:
            # TODO(阶段3/4): 掩码半透明填充，直观展示缺陷实际形状
            pass
    return canvas


def draw_stats(image, result, position: Optional[tuple] = None):
    """在图像左上角绘制统计信息（缺陷总数 / 良率 / 判定结果）。"""
    canvas = image.copy()
    text = f'defects={result.total_defects} {"PASS" if result.is_pass else "NG"}'
    origin = position or (10, 30)
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0) if result.is_pass else (0, 0, 255), 2)
    return canvas
