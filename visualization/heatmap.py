"""V1.1 扩展 02：检测结果空间密度热力图（零模型依赖，确定性）。

以每条 ``Detection`` 的 bbox 中心为脉冲点叠加高斯核，归一化后套用 JET
伪彩色（低=蓝 / 高=红），再与原图按 ``alpha`` 半透明融合后落盘。

该模块只消费 ``Frame`` + ``List[Detection]``，不修改 ``Detection`` 字段，
也不依赖 YOLO / CUDA，可作为 EigenCAM 不可用时的确定性兜底产物。
"""

import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from core import Detection, Frame

logger = logging.getLogger("defect.visual")

DEFAULT_SIGMA_RATIO = 0.05  # 高斯核 sigma = max(2, ratio × min(W, H))
DEFAULT_ALPHA = 0.6  # 热力图叠加透明度


def generate_density_heatmap(
    frame: Frame,
    detections: List[Detection],
    output_path: str | Path,
    *,
    sigma_ratio: float = DEFAULT_SIGMA_RATIO,
    alpha: float = DEFAULT_ALPHA,
) -> Optional[str]:
    """生成检测密度热力图并与原图融合落盘。

    :param frame: 统一帧对象，``image`` 为 BGR uint8
    :param detections: 检测结果列表；为空时直接返回 ``None``、不写文件
    :param output_path: 输出图片路径，父目录自动创建
    :return: 实际写入的路径；``detections`` 为空时返回 ``None``
    :raises ValueError: 帧非法或参数越界
    :raises OSError: 落盘失败
    """
    if frame is None or frame.image is None or frame.image.size == 0:
        raise ValueError("frame 非法：图像为空")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha 越界：{alpha!r}，应在 [0, 1] 区间")
    if sigma_ratio <= 0:
        raise ValueError(f"sigma_ratio 非法：{sigma_ratio!r}，应为正数")

    if not detections:
        logger.info("无检测结果，跳过密度热力图：%s", output_path)
        return None

    image = frame.image
    height, width = image.shape[:2]
    sigma = max(2.0, sigma_ratio * min(width, height))

    # 以 bbox 中心为脉冲点累加，再整体高斯模糊得到连续密度场
    impulses = np.zeros((height, width), dtype=np.float32)
    for detection in detections:
        x, y, w, h = detection.bbox
        cx = int(np.clip(x + w / 2.0, 0, width - 1))
        cy = int(np.clip(y + h / 2.0, 0, height - 1))
        impulses[cy, cx] += 1.0

    density = cv2.GaussianBlur(impulses, (0, 0), sigmaX=sigma, sigmaY=sigma)
    peak = float(density.max())
    if peak <= 0.0:  # 理论上不可达（detections 非空），防御性返回
        logger.warning("密度场全零，跳过热力图：%s", output_path)
        return None

    normalized = np.clip(density / peak * 255.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)  # 低=蓝，高=红

    # 按密度加权融合：密度为 0 处保留原图，峰值处达到设定 alpha（约 0.6）
    # 指数 0.6 < 1 用于抬升中低密度区的着色，避免只有红心可见
    weight = (normalized.astype(np.float32) / 255.0) ** 0.6
    alpha_map = (weight * alpha)[..., None]
    blended = (
        heat.astype(np.float32) * alpha_map + image.astype(np.float32) * (1.0 - alpha_map)
    ).astype(np.uint8)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), blended):
        raise OSError(f"密度热力图写入失败：{output}")

    logger.info("密度热力图已保存：%s（%d 条检测，sigma=%.1f）", output, len(detections), sigma)
    return str(output)
