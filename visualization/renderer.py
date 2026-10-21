"""阶段 5 §4.1 / §4.4 / §5.1：无头可视化（叠绘 + 落盘），不创建任何窗口。

- 输入 ``Frame`` + ``List[Detection]`` + ``Verdict``，输出带标注的 BGR 图像。
- 颜色/线宽/字号严格按 §4.1：四类 BGR 色表、按分辨率缩放的线宽与字号、
  左上角 PASS/FAIL 横幅（空检测列表同样绘制 PASS）。
- 异常语义见 §5.1：非法入参抛 ``ValueError``，落盘失败抛 ``OSError``。
"""

import logging
from pathlib import Path
from typing import List

import cv2
import numpy as np

from core import Detection, Frame, Verdict

logger = logging.getLogger("defect.visual")

# §4.1 四类缺陷固定配色（BGR，不随配置变化）
COLORS = {
    0: (0, 0, 255),  # scratch 红
    1: (255, 0, 0),  # dent 蓝
    2: (0, 255, 0),  # stain 绿
    3: (0, 165, 255),  # notch 橙
}
UNKNOWN_COLOR = (255, 255, 255)  # 越界 class_id 降级为白色
PASS_COLOR = (0, 200, 0)  # 良品横幅：绿
FAIL_COLOR = (0, 0, 255)  # 次品横幅：红
MASK_ALPHA = 0.35  # 掩码半透明填充
BANNER_ALPHA = 0.35  # 横幅底条透明度
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _thickness(width: int, height: int) -> int:
    """§4.1 线宽：max(1, round(min(W, H) / 400))。"""
    return max(1, round(min(width, height) / 400))


def _font_scale(width: int, height: int) -> float:
    """§4.1 标签字号：max(0.4, min(W, H) / 1000)。"""
    return max(0.4, min(width, height) / 1000)


def _banner_scale(width: int, height: int) -> float:
    """§4.1 PASS/FAIL 横幅字号：max(0.8, min(W, H) / 500)。"""
    return max(0.8, min(width, height) / 500)


def _validate(frame: Frame, detections: List[Detection], verdict: Verdict) -> np.ndarray:
    """按 §5.1 / §5.2 校验入参；返回图像引用。"""
    if not isinstance(frame, Frame):
        raise ValueError(f"frame 必须是 Frame，收到 {type(frame).__name__}")
    image = frame.image
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("frame.image 必须是 np.ndarray（空帧不允许进入可视化）")
    if image.size == 0:
        raise ValueError("frame.image 为空（size=0）")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"frame.image 必须是 H×W×3 的 BGR 图像，收到 shape={image.shape}")
    if frame.width != image.shape[1] or frame.height != image.shape[0]:
        raise ValueError(
            f"frame 宽高与图像不一致：width={frame.width}, height={frame.height}, "
            f"image.shape={image.shape}"
        )
    if detections is None or not isinstance(detections, list):
        raise ValueError(f"detections 必须是 list，收到 {type(detections).__name__}")
    for index, detection in enumerate(detections):
        if not isinstance(detection, Detection):
            raise ValueError(f"detections[{index}] 不是 Detection：{type(detection).__name__}")
    if not isinstance(verdict, Verdict):
        raise ValueError(f"verdict 必须是 Verdict，收到 {type(verdict).__name__}")
    return image


def _draw_mask(image: np.ndarray, detection: Detection, color: tuple) -> None:
    """掩码半透明填充；shape 不匹配时只记 warning（§5.2）。"""
    mask = detection.mask
    if mask is None:
        return
    height, width = image.shape[0], image.shape[1]
    if not isinstance(mask, np.ndarray) or mask.shape[:2] != (height, width):
        logger.warning("class_id=%d 的 mask 尺寸与图像不一致，跳过填充", detection.class_id)
        return
    binary = mask if mask.dtype == bool else mask > 0
    if not binary.any():
        return
    tint = np.asarray(color, dtype=np.float32)
    image[binary] = (
        image[binary].astype(np.float32) * (1.0 - MASK_ALPHA) + tint * MASK_ALPHA
    ).astype(np.uint8)


def _draw_detection(image: np.ndarray, detection: Detection) -> None:
    """单条检测：clamp 边界后画框 + 标签；完全越界/退化框跳过并记 warning。"""
    height, width = image.shape[0], image.shape[1]
    x, y, w, h = (int(value) for value in detection.bbox)
    if w <= 0 or h <= 0:
        logger.warning("跳过退化 bbox（w=%d, h=%d）：class_id=%d", w, h, detection.class_id)
        return

    x1, y1 = max(0, min(width, x)), max(0, min(height, y))
    x2, y2 = max(0, min(width, x + w)), max(0, min(height, y + h))
    if x2 <= x1 or y2 <= y1:  # 完全越界
        logger.warning("跳过完全越界的 bbox=%s：class_id=%d", detection.bbox, detection.class_id)
        return

    color = COLORS.get(detection.class_id)
    if color is None:
        logger.warning("未知 class_id=%d，使用白色绘制", detection.class_id)
        color = UNKNOWN_COLOR

    _draw_mask(image, detection, color)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, _thickness(width, height))

    label = f"{detection.class_name} {detection.confidence:.2f}"
    font_scale = _font_scale(width, height)
    (text_w, text_h), baseline = cv2.getTextSize(label, FONT, font_scale, 1)
    top = max(y1, text_h + baseline)  # 标签贴在框上沿，越界时下移
    left = max(0, min(width - text_w, x1))
    cv2.rectangle(image, (left, top - text_h - baseline), (left + text_w, top), color, -1)
    cv2.putText(image, label, (left, top - baseline), FONT, font_scale, (255, 255, 255), 1, cv2.LINE_8)


def _draw_banner(image: np.ndarray, verdict: Verdict) -> None:
    """§4.1 左上角 PASS/FAIL 横幅；只由 verdict.is_pass 决定，不看检测条数。"""
    height, width = image.shape[0], image.shape[1]
    text = "PASS" if verdict.is_pass else "FAIL"
    color = PASS_COLOR if verdict.is_pass else FAIL_COLOR
    scale = _banner_scale(width, height)
    (text_w, text_h), baseline = cv2.getTextSize(text, FONT, scale, 2)

    margin = max(2, round(min(width, height) / 100))
    bar_w = min(width, text_w + 2 * margin)
    bar_h = min(height, text_h + baseline + 2 * margin)
    if bar_w > 0 and bar_h > 0:
        roi = image[0:bar_h, 0:bar_w]
        overlay = np.empty_like(roi)
        overlay[:] = color
        cv2.addWeighted(overlay, BANNER_ALPHA, roi, 1.0 - BANNER_ALPHA, 0, roi)
    cv2.putText(
        image,
        text,
        (margin, margin + text_h),
        FONT,
        scale,
        color,
        2,
        cv2.LINE_8,
    )


def render(frame: Frame, detections: List[Detection], verdict: Verdict) -> np.ndarray:
    """叠加检测框与判定横幅，返回新图像（不修改入参 ``frame.image``）。"""
    source = _validate(frame, detections, verdict)
    image = source.copy()
    for detection in detections:
        _draw_detection(image, detection)
    _draw_banner(image, verdict)
    return image


def render_to_file(
    frame: Frame, detections: List[Detection], verdict: Verdict, output_path: str
) -> str:
    """渲染并落盘为图片；返回写入路径。目录不存在自动创建，写失败抛 ``OSError``。"""
    if not isinstance(output_path, str) or not output_path:
        raise ValueError(f"output_path 必须是非空 str，收到 {output_path!r}")
    image = render(frame, detections, verdict)
    path = Path(output_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(path), image)
    except OSError:
        raise
    except Exception as exc:  # cv2 编码异常统一转为 OSError（§5.1）
        raise OSError(f"写入图片失败：{path}：{exc}") from exc
    if not ok:
        raise OSError(f"写入图片失败（cv2.imwrite 返回 False）：{path}")
    logger.debug("已写入标注图：%s", path)
    return str(path)


__all__ = ["COLORS", "render", "render_to_file"]
