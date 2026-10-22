"""V1.1 扩展 02 第二步：EigenCAM 模型注意力热力图（梯度无关）。

流程：在实际参与前向的 backbone 层（默认 idx 6 / stage-4 C2f）注册 forward hook →
复用 ``detector.detect()`` 完成既有预处理与推理 → 取激活 ``[1, C, H, W]`` →
按通道维做 PCA 第一主成分（EigenCAM）→ 归一化 → 上采样到原图 →
JET 伪彩色 → 复用 ``heatmap.blend_heatmap`` 按密度加权融合落盘。

该模块只读模型、不改 ``Detection`` 字段，也不改变 ``detect()`` 签名。
"""

import logging
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np

from core import Frame
from utils.exceptions import DetectorInferenceError
from visualization.heatmap import DEFAULT_ALPHA, blend_heatmap

logger = logging.getLogger("defect.visual")

DEFAULT_LAYER_INDEX = 6  # YOLOv8m backbone stage-4 C2f，输出 [1, 384, 40, 40]
DEFAULT_LAYER_DESC = "默认使用 backbone stage-4 C2f（idx 6, 40×40）"
PCA_VARIANCE_EPS = 1e-8  # 第一主成分奇异值低于此值视为退化
CAM_PTP_EPS = 1e-6  # 归一化后动态范围低于此值视为全黑/全白


def _resolve_layer(model, layer_name: Optional[Union[int, str]]):
    """按索引或名称解析目标层；不存在抛 ``DetectorInferenceError``。"""
    layers = getattr(model, "model", None)
    if layers is None or not hasattr(layers, "__getitem__"):
        raise DetectorInferenceError("模型缺少 .model 层序列，无法定位 EigenCAM 目标层")

    if layer_name is None:
        index = DEFAULT_LAYER_INDEX
    elif isinstance(layer_name, int):
        index = layer_name
    elif isinstance(layer_name, str) and layer_name.isdigit():
        index = int(layer_name)
    else:
        raise DetectorInferenceError(f"layer_name 非法：{layer_name!r}，应为层索引（如 9）")

    total = len(layers)
    if not -total <= index < total:
        raise DetectorInferenceError(f"层索引越界：{index}，当前模型共 {total} 层")
    return layers[index], index


def _eigencam(activation: np.ndarray) -> tuple[np.ndarray, Optional[str]]:
    """激活 [C, H, W] → CAM [H, W]；退化时降级为通道均值并返回原因。"""
    channels, height, width = activation.shape
    flattened = activation.reshape(channels, -1).astype(np.float32)
    flattened -= flattened.mean(axis=1, keepdims=True)  # 逐通道去均值

    reason = None
    cam = None
    try:
        # 通道为样本、空间位置为特征：取左奇异向量第一列作为通道权重
        left, singular, _ = np.linalg.svd(flattened, full_matrices=False)
        if singular.size and singular[0] > PCA_VARIANCE_EPS:
            cam = (left[:, 0] @ flattened).reshape(height, width)
        else:
            reason = f"PCA 第一主成分方差≈0（sigma={singular[0] if singular.size else 0.0:.2e}）"
    except np.linalg.LinAlgError as exc:
        reason = f"PCA 分解失败（{exc}）"

    if cam is None or float(np.ptp(cam)) < CAM_PTP_EPS:
        reason = reason or "PCA 结果全黑/全白"
        cam = activation.mean(axis=0)
        if float(np.ptp(cam)) < CAM_PTP_EPS:
            raise DetectorInferenceError("EigenCAM 退化：PCA 与通道均值均无有效动态范围")
        return cam, reason
    return cam, None


def _normalize_cam(
    cam: np.ndarray, clip_percentile: Optional[tuple[float, float]]
) -> Optional[np.ndarray]:
    """分位裁剪后 min-max 到 [0, 1]；分位区间退化时退回绝对 min-max。"""
    if clip_percentile is not None:
        low, high = np.percentile(cam, clip_percentile)
        if high - low >= CAM_PTP_EPS:
            cam = np.clip(cam, low, high)
        else:  # 分布几乎全平：不做分位裁剪，避免把噪声拉伸放大
            logger.info("分位区间退化（%.2e），退回绝对 min-max：%s", high - low, clip_percentile)
    cam = cam - cam.min()
    span = float(cam.max())
    if span <= 0.0:
        return None
    return cam / span


def _forward_model(detector, frame: Frame, base_model):
    """返回实际参与前向的模型对象。

    ultralytics 的 ``AutoBackend`` 会在参数处于 inference tensor 状态时 deepcopy
    模型，此后 ``predict`` 一直复用该副本；hook 若挂在加载时的原模型上，从第二次
    推理起不会触发。因此优先取 ``detector.inference_model``，并在 predictor 尚未
    创建（首次调用）时先做一次丢弃结果的 warmup ``detect()``。
    """
    model = getattr(detector, "inference_model", None)
    if model is None:  # 未提供该只读属性：退回加载时的模型
        return base_model
    if model is base_model:  # predictor 尚未创建：跑一次 warmup 让副本就位
        try:
            detector.detect(frame)
        except DetectorInferenceError:
            raise
        except Exception as exc:
            raise DetectorInferenceError(f"EigenCAM 预热推理失败：{exc}") from exc
        model = getattr(detector, "inference_model", None) or base_model
    return model


def generate_eigencam_heatmap(
    frame: Frame,
    detector,
    output_path: str | Path,
    layer_name: Optional[Union[int, str]] = None,
    *,
    alpha: float = DEFAULT_ALPHA,
    clip_percentile: Optional[tuple[float, float]] = (1.0, 99.0),
) -> str:
    """生成 EigenCAM 注意力热力图并与原图融合落盘。

    :param frame: 统一帧对象，``image`` 为 BGR uint8
    :param detector: 已 ``load`` 的 YOLO 检测器（需暴露只读 ``model``；提供
        ``inference_model`` 时 hook 会挂到实际前向的副本上）
    :param output_path: 输出图片路径，父目录自动创建
    :param layer_name: 目标层索引（int 或数字字符串）；``None`` 用默认第 6 层
    :param alpha: 峰值处的叠加透明度
    :param clip_percentile: 分位归一化上下界 ``(low, high)``；``None`` 或区间退化时
        退回绝对 min-max。默认 ``(1.0, 99.0)`` 抑制高密度样本的饱和
    :return: 实际写入的路径
    :raises ValueError: 帧非法或分位参数越界
    :raises DetectorInferenceError: 模型未加载、层不存在、hook 失败或推理失败
    """
    if frame is None or frame.image is None or frame.image.size == 0:
        raise ValueError("frame 非法：图像为空")
    if clip_percentile is not None:
        low, high = clip_percentile
        if not 0.0 <= low < high <= 100.0:
            raise ValueError(f"clip_percentile 非法：{clip_percentile!r}，需满足 0≤low<high≤100")

    base_model = getattr(detector, "model", None)
    if base_model is None:
        raise DetectorInferenceError("检测器模型未加载：请先调用 load(cfg) 再生成注意力热力图")

    model = _forward_model(detector, frame, base_model)
    target, index = _resolve_layer(model, layer_name)
    if layer_name is None:
        logger.info(DEFAULT_LAYER_DESC)

    try:
        import torch
    except ImportError as exc:  # 预留环境不含 torch 时明确报错
        raise DetectorInferenceError(f"EigenCAM 需要 torch，导入失败：{exc}") from exc

    captured: List[np.ndarray] = []

    def _hook(_module, _inputs, output) -> None:
        tensor = output[0] if isinstance(output, (list, tuple)) else output
        if torch.is_tensor(tensor) and tensor.dim() == 4:
            captured.append(tensor.detach().float().cpu().numpy())

    try:
        handle = target.register_forward_hook(_hook)
    except Exception as exc:  # 注册失败（层类型不支持等）
        raise DetectorInferenceError(f"forward hook 注册失败（层 {index}）：{exc}") from exc

    try:
        model.eval()
        with torch.no_grad():
            detector.detect(frame)  # 复用既有预处理 / letterbox / 设备调度
    except DetectorInferenceError:
        raise
    except Exception as exc:
        raise DetectorInferenceError(f"EigenCAM 前向推理失败（层 {index}）：{exc}") from exc
    finally:
        handle.remove()  # 异常路径同样摘除，避免污染后续 detect()

    if not captured:
        raise DetectorInferenceError(f"未捕获到层 {index} 的激活，请检查层索引是否对应卷积输出")
    activation = captured[-1][0]  # [C, H, W]
    if activation.ndim != 3:
        raise DetectorInferenceError(f"层 {index} 激活形状异常：{activation.shape}")

    cam, degrade_reason = _eigencam(activation)
    if degrade_reason:
        logger.warning("EigenCAM 降级为通道均值：%s", degrade_reason)

    cam = _normalize_cam(cam, clip_percentile)
    if cam is None:
        raise DetectorInferenceError("EigenCAM 归一化失败：注意力图动态范围为 0")

    image = frame.image
    height, width = image.shape[:2]
    upsampled = cv2.resize(cam, (width, height), interpolation=cv2.INTER_LINEAR)
    normalized = np.clip(upsampled * 255.0, 0, 255).astype(np.uint8)
    blended = blend_heatmap(image, normalized, alpha=alpha)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), blended):
        raise OSError(f"EigenCAM 热力图写入失败：{output}")

    logger.info(
        "EigenCAM 热力图已保存：%s（层 %d，激活 %s，%s）",
        output,
        index,
        tuple(activation.shape),
        "已降级" if degrade_reason else "PCA 正常",
    )
    return str(output)
