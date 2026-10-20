"""阶段 4 §5：YoloDetector —— YOLO 系列深度学习检测器。

- 实现阶段 0 §5.4.2 的 ``Detector`` 契约（``name`` / ``version`` / ``load`` /
  ``detect`` / ``release``），签名与阶段 0 逐参数一致。
- 输出严格符合阶段 0 §5.4.1 的 ``Detection`` 结构，字段转换规则见阶段 4 §4.2。
- ``class_id`` 复用阶段 3 §4.1 映射表（只复用、不重定义）；参数表见 §4.4，
  异常语义见 §5.1，资源生命周期见 §5.2。
- torch / ultralytics 为可选依赖：仅在 ``load`` 时导入，未安装即抛
  ``DetectorLoadError``，不阻塞 opencv 路线的正常使用。
"""

import gc
import logging
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from core import Detector, Detection, Frame
from detectors.opencv_detector import CLASS_MAP  # 阶段 3 §4.1 唯一权威映射表
from utils.exceptions import DetectorInferenceError, DetectorLoadError

# --------------------------------------------------------------------------- #
# §4.4 参数配置（配置驱动；禁止在算法里写死阈值）
# --------------------------------------------------------------------------- #


def _require_num(name: str, value: object, low: float = 0.0, high: float = 1.0) -> None:
    """数值校验：拒绝 bool（bool 是 int 子类）与越界值。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DetectorLoadError(f"{name} 必须是数值，收到 {value!r}")
    if value < low or value > high:
        raise DetectorLoadError(f"{name} 必须落在 [{low}, {high}]，收到 {value}")


def _require_int(name: str, value: object, low: int = 1) -> None:
    """正整数校验：拒绝 bool 与非正整数。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise DetectorLoadError(f"{name} 必须是 int，收到 {value!r}")
    if value < low:
        raise DetectorLoadError(f"{name} 必须 >= {low}，收到 {value}")


def _require_device(name: str, value: object) -> None:
    """设备语法校验（可用性校验在 ``load`` 中做，见 §5.1）。"""
    if not isinstance(value, str) or not value:
        raise DetectorLoadError(f"{name} 必须是非空 str，收到 {value!r}")
    if value == "cpu":
        return
    if value == "cuda":
        return
    if value.startswith("cuda:"):
        index = value.split(":", 1)[1]
        if index.isdigit():
            return
    raise DetectorLoadError(f"{name} 只支持 'cpu' / 'cuda' / 'cuda:N'（N >= 0），收到 {value!r}")


@dataclass
class YoloDetectorConfig:
    """§4.4 参数表；``model_path`` 必填，其余字段缺省即表内默认值。"""

    model_path: str
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    device: str = "cpu"
    imgsz: int = 640

    @classmethod
    def from_dict(cls, params: dict) -> "YoloDetectorConfig":
        """按 §4.4 校验并构造；缺必填项/未知字段/非法值均抛 ``DetectorLoadError``。"""
        if not isinstance(params, dict):
            raise DetectorLoadError(f"detector.params 必须是 dict，收到 {type(params).__name__}")
        unknown = sorted(set(params) - set(PARAM_NAMES))
        if unknown:
            raise DetectorLoadError(f"detector.params 含未知字段 {unknown}；允许字段见阶段 4 §4.4")
        if "model_path" not in params:
            raise DetectorLoadError("detector.params 缺少必填字段 model_path（阶段 4 §4.4）")
        cfg = cls(**params)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        """逐项校验（§4.4 校验规则）；任一不合法即抛 ``DetectorLoadError``。"""
        if not isinstance(self.model_path, str) or not self.model_path:
            raise DetectorLoadError(f"model_path 必须是非空 str，收到 {self.model_path!r}")
        path = Path(self.model_path)
        if not path.is_file():
            raise DetectorLoadError(f"model_path 指向的文件不存在：{self.model_path}")
        if not os.access(path, os.R_OK):
            raise DetectorLoadError(f"model_path 指向的文件不可读：{self.model_path}")

        _require_num("conf_threshold", self.conf_threshold)
        _require_num("iou_threshold", self.iou_threshold)
        _require_device("device", self.device)
        _require_int("imgsz", self.imgsz)

    @property
    def uses_cuda(self) -> bool:
        """是否要求使用 CUDA 设备（含 ``cuda`` 与 ``cuda:N``）。"""
        return self.device == "cuda" or self.device.startswith("cuda:")

    @property
    def cuda_index(self) -> Optional[int]:
        """``cuda:N`` 中的设备号 N；``cuda`` / ``cpu`` 返回 ``None``。"""
        if self.device.startswith("cuda:"):
            return int(self.device.split(":", 1)[1])
        return None


PARAM_NAMES: Tuple[str, ...] = tuple(f.name for f in fields(YoloDetectorConfig))


# --------------------------------------------------------------------------- #
# §5 检测器实现
# --------------------------------------------------------------------------- #
class YoloDetector(Detector):
    """YOLO 系列检测器：load 加载权重到 device，detect 推理并转 Detection。"""

    def __init__(self) -> None:
        self._cfg: Optional[YoloDetectorConfig] = None
        self._model = None
        self._logger = logging.getLogger("defect.yolo")

    @property
    def name(self) -> str:
        """模型变体标识（如 ``yolov8n``）；未 ``load`` 时回退为 ``"yolo"``。"""
        return Path(self._cfg.model_path).stem if self._cfg is not None else "yolo"

    @property
    def version(self) -> str:
        return "0.1.0"

    def load(self, cfg: dict) -> None:
        """读入 §4.4 参数并加载权重到指定 device；失败抛 ``DetectorLoadError``。"""
        if not isinstance(cfg, dict):
            raise DetectorLoadError(f"detector 配置必须是 dict，收到 {type(cfg).__name__}")
        if "params" not in cfg:
            raise DetectorLoadError("detector 配置缺少 params 字段（字段表见阶段 4 §4.4）")

        config = YoloDetectorConfig.from_dict(cfg["params"])

        torch = self._import_torch()
        if config.uses_cuda:
            if not torch.cuda.is_available():
                raise DetectorLoadError(
                    f"device={config.device!r} 要求 CUDA，但当前环境 CUDA 不可用："
                    "不静默回退到 cpu（阶段 4 §4.4 / §5.1）"
                )
            index = config.cuda_index
            if index is not None and index >= torch.cuda.device_count():
                raise DetectorLoadError(
                    f"device={config.device!r} 的设备号越界：可用设备数 {torch.cuda.device_count()}"
                )

        model = self._build_model(config)
        try:
            if hasattr(model, "to"):
                model.to(config.device)  # 权重落到目标设备（cuda 时占显存）
            if hasattr(model, "eval"):
                model.eval()
        except Exception as exc:  # 半初始化失败：清理后再抛（§5.2 异常路径释放）
            self._cleanup(model, config)
            raise DetectorLoadError(f"模型加载到 {config.device} 失败：{exc}") from exc

        self._model = model
        self._cfg = config
        self._logger.info(
            "已加载 YOLO 模型：%s（device=%s, conf=%.2f, iou=%.2f, imgsz=%d）",
            config.model_path,
            config.device,
            config.conf_threshold,
            config.iou_threshold,
            config.imgsz,
        )
        return None

    def detect(self, frame: Frame) -> List[Detection]:
        """检测一帧；无缺陷返回 ``[]``，输入非法抛 ``ValueError``，推理失败抛 ``DetectorInferenceError``。"""
        image = self._validate_frame(frame)
        model, config = self._model, self._cfg
        if model is None or config is None:
            raise DetectorInferenceError("模型未加载：请先调用 load(cfg)；release() 之后需重新 load")

        try:
            results = model.predict(
                source=image,
                conf=config.conf_threshold,
                iou=config.iou_threshold,
                imgsz=config.imgsz,
                device=config.device,
                verbose=False,
            )
        except Exception as exc:  # 含 CUDA OOM / RuntimeError 等（§5.1）
            raise DetectorInferenceError(f"YOLO 推理失败：{exc}") from exc

        height, width = int(image.shape[0]), int(image.shape[1])
        detections = self._to_detections(results, width, height)
        self._logger.debug("帧 %d：输出 %d 条", frame.frame_index, len(detections))
        return detections

    def release(self) -> None:
        """释放模型与显存（§5.2）；幂等，重复调用直接返回。"""
        if self._model is None:  # 未加载或已释放：无操作，保证幂等
            return
        uses_cuda = self._cfg is not None and self._cfg.uses_cuda
        self._model = None
        gc.collect()
        if uses_cuda:
            try:
                import torch

                torch.cuda.empty_cache()
            except ImportError:  # 依赖已卸载：无显存可释放
                pass
        self._logger.info("已释放 YOLO 模型与显存缓存")

    # ---------- §4.2 推理结果 → Detection 转换 ----------

    def _to_detections(self, results, width: int, height: int) -> List[Detection]:
        """按 §4.2 字段转换表逐框转换；class_id 越界/退化框丢弃并记 warning。"""
        detections: List[Detection] = []
        area_total = float(width * height)  # 空帧已在入口抛 ValueError，恒 > 0
        for result in results or []:
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            for xyxy, conf, cls in zip(
                self._to_numpy(boxes.xyxy), self._to_numpy(boxes.conf), self._to_numpy(boxes.cls)
            ):
                class_id = int(cls)
                if class_id not in CLASS_MAP:  # 越界不中断整帧，仅丢弃该条
                    self._logger.warning(
                        "丢弃 class_id=%d：不在阶段 3 §4.1 映射表 %s 内", class_id, sorted(CLASS_MAP)
                    )
                    continue

                x1, y1, x2, y2 = (int(round(float(value))) for value in xyxy)
                x, y = max(0, min(width, x1)), max(0, min(height, y1))
                w = max(0, min(width, x2)) - x  # clamp 到图像边界 [0, W]×[0, H]
                h = max(0, min(height, y2)) - y
                if w <= 0 or h <= 0:
                    self._logger.warning("丢弃退化框：clamp 后 w=%d, h=%d", w, h)
                    continue

                area_px = w * h  # 框级检测无掩码，以 bbox 面积近似（§4.2 / §8 特殊风险）
                detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=CLASS_MAP[class_id],
                        bbox=(x, y, w, h),
                        area_px=area_px,
                        area_ratio=area_px / area_total,
                        confidence=float(np.clip(float(conf), 0.0, 1.0)),
                        mask=None,
                        severity=None,
                    )
                )
        return detections

    @staticmethod
    def _to_numpy(tensor) -> np.ndarray:
        """张量/数组统一转 numpy（推理输出可能在 GPU 上）。"""
        if hasattr(tensor, "detach"):
            return tensor.detach().cpu().numpy()
        return np.asarray(tensor)

    # ---------- 依赖与模型构造 ----------

    @staticmethod
    def _import_torch():
        try:
            import torch
        except ImportError as exc:
            raise DetectorLoadError(
                "未安装 torch，无法使用 YOLO 检测器；请先 `pip install torch` 或改用 opencv"
            ) from exc
        return torch

    @staticmethod
    def _build_model(config: YoloDetectorConfig):
        """构造 YOLO 模型；依赖缺失或权重格式错误一律转 ``DetectorLoadError``。"""
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise DetectorLoadError(
                "未安装 ultralytics，无法使用 YOLO 检测器；请先 `pip install ultralytics` 或改用 opencv"
            ) from exc
        try:
            return YOLO(config.model_path)
        except Exception as exc:  # 反序列化/格式错误等（§5.1）
            raise DetectorLoadError(f"模型加载失败（文件损坏或格式错误）：{config.model_path}") from exc

    def _cleanup(self, model, config: YoloDetectorConfig) -> None:
        """load 中途失败时的半初始化清理（§5.2 异常路径释放）。"""
        del model
        gc.collect()
        if config.uses_cuda:
            try:
                import torch

                torch.cuda.empty_cache()
            except ImportError:
                pass

    # ---------- 输入校验（§5.1，语义与阶段 3 §6.1 一致）----------

    @staticmethod
    def _validate_frame(frame: Frame) -> np.ndarray:
        if frame is None:
            raise ValueError("frame 不可为 None")
        image = getattr(frame, "image", None)
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("frame.image 必须是非空 np.ndarray")
        if image.ndim < 2:
            raise ValueError(f"frame.image 至少 2 维，收到 {image.ndim} 维")
        height, width = int(image.shape[0]), int(image.shape[1])
        if height == 0 or width == 0:
            raise ValueError(f"frame.image 尺寸为 0，属非法输入：{width}x{height}")
        if height != int(frame.height) or width != int(frame.width):
            raise ValueError(
                f"frame.image 尺寸 {width}x{height} 与 frame.width/height "
                f"{frame.width}x{frame.height} 不一致"
            )
        return image
