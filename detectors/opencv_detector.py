"""阶段 3 §6：OpencvDetector —— 传统 OpenCV 四类缺陷检测器。

- 实现阶段 0 §5.4.2 的 ``Detector`` 契约（``name`` / ``version`` / ``load`` /
  ``detect`` / ``release``），签名与阶段 0 逐参数一致。
- 输出严格符合阶段 0 §5.4.1 的 ``Detection`` 结构，字段口径见阶段 3 §4.3。
- 类别映射表见 §4.1、置信度规则见 §4.2、参数表见 §4.4、异常语义见 §6.1。
"""

import logging
from dataclasses import dataclass, fields
from typing import Dict, List, Tuple

import cv2
import numpy as np

from core import Detector, Detection, Frame
from utils.exceptions import DetectorInferenceError, DetectorLoadError

# --------------------------------------------------------------------------- #
# §4.1 类别映射表（写死；阶段 4/5/7 复用同一张表，不得各自定义）
# --------------------------------------------------------------------------- #
CLASS_MAP: Dict[int, str] = {
    0: "scratch",  # 划痕
    1: "dent",  # 凹坑
    2: "stain",  # 污渍
    3: "notch",  # 缺口
}

#: Lab 三通道取值 0~255，欧氏色距理论上限（§4.2 污渍 score 的归一化分母）
LAB_DIST_MAX = float(np.sqrt(3.0) * 255.0)

#: 连通域/形态学通用 3×3 椭圆核（结构元素尺寸，非阈值参数）
_K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

#: 结果收集处的硬性过滤上限：bbox 矩形面积 / 整图面积，超过即丢弃
MAX_BBOX_AREA_RATIO = 0.15


# --------------------------------------------------------------------------- #
# §4.4 参数配置（配置驱动；禁止在算法里写死阈值）
# --------------------------------------------------------------------------- #
def _require_int(name: str, value: object, low: int = 0, high: int = 255) -> None:
    """整型校验：拒绝 bool（bool 是 int 子类）与越界值。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise DetectorLoadError(f"{name} 必须是 int，收到 {value!r}")
    if not low <= value <= high:
        raise DetectorLoadError(f"{name} 必须落在 [{low}, {high}]，收到 {value}")


def _require_num(
    name: str,
    value: object,
    low: float = 0.0,
    high: float = 1.0,
    low_strict: bool = False,
) -> None:
    """数值校验：low_strict=True 时取严格大于 low。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DetectorLoadError(f"{name} 必须是数值，收到 {value!r}")
    if (value <= low) if low_strict else (value < low):
        raise DetectorLoadError(f"{name} 必须 {'>' if low_strict else '>='} {low}，收到 {value}")
    if value > high:
        raise DetectorLoadError(f"{name} 必须 <= {high}，收到 {value}")


@dataclass
class OpencvDetectorConfig:
    """§4.4 参数表；缺省值即表内默认值。"""

    min_area: int = 10
    min_confidence: float = 0.05
    denoise: str = "gaussian"
    denoise_kernel: int = 5
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    scratch_canny_low: int = 50
    scratch_canny_high: int = 150
    scratch_min_aspect: float = 3.0
    scratch_max_width: int = 8
    dent_log_sigma: float = 3.0
    dent_roundness_min: float = 0.6
    dent_radius_min: int = 2
    dent_radius_max: int = 20
    stain_sat_min: int = 60
    stain_color_dist_min: float = 30.0
    stain_area_ratio_min: float = 0.001
    notch_convexity_min: float = 0.02

    @classmethod
    def from_dict(cls, params: dict) -> "OpencvDetectorConfig":
        """按 §4.4 校验并构造；字段缺失用默认值，未知字段/非法值抛 DetectorLoadError。"""
        if not isinstance(params, dict):
            raise DetectorLoadError(f"detector.params 必须是 dict，收到 {type(params).__name__}")
        unknown = sorted(set(params) - set(PARAM_NAMES))
        if unknown:
            raise DetectorLoadError(f"detector.params 含未知字段 {unknown}；允许字段见阶段 3 §4.4")
        cfg = cls(**params)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        """逐项校验（§4.4 校验规则）；任一不合法即抛 DetectorLoadError。"""
        _require_int("min_area", self.min_area, low=1, high=2**31 - 1)
        _require_num("min_confidence", self.min_confidence)

        if self.denoise not in ("gaussian", "median"):
            raise DetectorLoadError(f"denoise 必须是 'gaussian' / 'median'，收到 {self.denoise!r}")
        _require_int("denoise_kernel", self.denoise_kernel, low=1, high=255)
        if self.denoise_kernel % 2 == 0:
            raise DetectorLoadError(f"denoise_kernel 必须是奇数，收到 {self.denoise_kernel}")

        _require_num("clahe_clip", self.clahe_clip, low=0.0, high=255.0, low_strict=True)
        _require_int("clahe_grid", self.clahe_grid, low=1, high=255)

        _require_int("scratch_canny_low", self.scratch_canny_low, low=0, high=65535)
        _require_int("scratch_canny_high", self.scratch_canny_high, low=0, high=65535)
        if self.scratch_canny_low > self.scratch_canny_high:
            raise DetectorLoadError(
                "scratch_canny_low 不得大于 scratch_canny_high："
                f"{self.scratch_canny_low} > {self.scratch_canny_high}"
            )
        _require_num("scratch_min_aspect", self.scratch_min_aspect, low=0.0, high=1000.0, low_strict=True)
        _require_int("scratch_max_width", self.scratch_max_width, low=1, high=65535)

        _require_num("dent_log_sigma", self.dent_log_sigma, low=0.0, high=255.0, low_strict=True)
        _require_num("dent_roundness_min", self.dent_roundness_min)
        _require_int("dent_radius_min", self.dent_radius_min, low=1, high=65535)
        _require_int("dent_radius_max", self.dent_radius_max, low=1, high=65535)
        if self.dent_radius_min > self.dent_radius_max:
            raise DetectorLoadError(
                f"dent_radius_min 不得大于 dent_radius_max：{self.dent_radius_min} > {self.dent_radius_max}"
            )

        _require_int("stain_sat_min", self.stain_sat_min, low=0, high=255)
        _require_num("stain_color_dist_min", self.stain_color_dist_min, low=0.0, high=LAB_DIST_MAX)
        _require_num("stain_area_ratio_min", self.stain_area_ratio_min)

        _require_num("notch_convexity_min", self.notch_convexity_min)


PARAM_NAMES: Tuple[str, ...] = tuple(f.name for f in fields(OpencvDetectorConfig))


@dataclass
class _Candidate:
    """内部候选：原始轮廓 + 规则分数（§4.2）+ 全图尺寸掩码（§4.3）。"""

    class_id: int
    contour: np.ndarray
    score: float
    mask: np.ndarray


# --------------------------------------------------------------------------- #
# §6 检测器实现
# --------------------------------------------------------------------------- #
class OpencvDetector(Detector):
    """传统 OpenCV 检测器：四类缺陷各跑通一条规则链路。"""

    def __init__(self) -> None:
        # 未调用 load() 时按 §4.4 表内默认值运行
        self._cfg = OpencvDetectorConfig()
        self._logger = logging.getLogger("defect.opencv")

    @property
    def name(self) -> str:
        return "opencv"

    @property
    def version(self) -> str:
        return "0.1.0"

    def load(self, cfg: dict) -> None:
        """读入 §4.4 参数；非法配置抛 DetectorLoadError（由调用方捕获）。"""
        if not isinstance(cfg, dict):
            raise DetectorLoadError(f"detector 配置必须是 dict，收到 {type(cfg).__name__}")
        if "params" not in cfg:
            raise DetectorLoadError("detector 配置缺少 params 字段（字段表见阶段 3 §4.4）")
        self._cfg = OpencvDetectorConfig.from_dict(cfg["params"])
        return None

    def detect(self, frame: Frame) -> List[Detection]:
        """检测一帧；无缺陷返回 []，输入非法抛 ValueError，内部异常抛 DetectorInferenceError。"""
        image = self._validate_frame(frame)
        cfg = self._cfg

        try:
            gray, enhanced = self._preprocess(image)
            candidates: List[_Candidate] = []
            candidates.extend(self._detect_scratch(enhanced))
            candidates.extend(self._detect_dent(gray))
            candidates.extend(self._detect_stain(image))
            candidates.extend(self._detect_notch(gray))
        except cv2.error as exc:  # OpenCV 内部调用异常（§6.1）
            raise DetectorInferenceError(f"OpenCV 内部异常：{exc}") from exc

        height, width = int(image.shape[0]), int(image.shape[1])
        detections: List[Detection] = []
        for candidate in candidates:
            area_px = int(round(float(cv2.contourArea(candidate.contour))))
            if area_px < cfg.min_area:  # §4.5 去噪下限：过小碎片直接丢弃
                continue
            confidence = float(np.clip(candidate.score, 0.0, 1.0))  # §4.2 恒非 None
            if confidence < cfg.min_confidence:
                continue
            x, y, w, h = (int(v) for v in cv2.boundingRect(candidate.contour))
            if w * h > MAX_BBOX_AREA_RATIO * width * height:  # 硬性过滤：整框过大直接丢弃
                continue
            detections.append(
                Detection(
                    class_id=candidate.class_id,
                    class_name=CLASS_MAP[candidate.class_id],
                    bbox=(x, y, w, h),
                    area_px=area_px,
                    area_ratio=area_px / float(width * height),
                    confidence=confidence,
                    mask=candidate.mask,
                    severity=None,  # 阶段 6 前恒为 None
                )
            )

        counted = {cid: sum(1 for d in detections if d.class_id == cid) for cid in CLASS_MAP}
        self._logger.debug(
            "帧 %d：输出 %d 条（划痕 %d / 凹坑 %d / 污渍 %d / 缺口 %d）",
            frame.frame_index,
            len(detections),
            counted[0],
            counted[1],
            counted[2],
            counted[3],
        )
        return detections

    def release(self) -> None:
        """无持久外部资源（无模型权重/文件/摄像头句柄），幂等空操作（§6.2）。"""
        return None

    # ---------- 输入校验（§6.1）----------

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

    # ---------- 预处理（§3 WBS 第 1 步）----------

    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """灰度化 → 去噪（高斯/中值）→ CLAHE 光照校正。"""
        cfg = self._cfg
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if cfg.denoise == "gaussian":
            gray = cv2.GaussianBlur(gray, (cfg.denoise_kernel, cfg.denoise_kernel), 0)
        else:
            gray = cv2.medianBlur(gray, cfg.denoise_kernel)
        clahe = cv2.createCLAHE(
            clipLimit=cfg.clahe_clip,
            tileGridSize=(cfg.clahe_grid, cfg.clahe_grid),
        )
        return gray, clahe.apply(gray)

    # ---------- 四类缺陷链路（§5）----------

    def _line_response(self, enhanced: np.ndarray) -> np.ndarray:
        """方向性形态学响应（§5 划痕：方向性滤波 + 顶帽/黑帽）。

        取水平/垂直线性核的顶帽与黑帽响应之最大值，以同时覆盖亮线/暗线两种极性；
        核长由 ``scratch_max_width`` 推导（2×最大线宽+1），不引入额外魔法数。
        """
        length = 2 * self._cfg.scratch_max_width + 1
        kernels = (
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, length)),
            cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1)),
        )
        responses = [
            cv2.morphologyEx(enhanced, operation, kernel)
            for operation in (cv2.MORPH_TOPHAT, cv2.MORPH_BLACKHAT)
            for kernel in kernels
        ]
        return np.maximum.reduce(responses)

    def _detect_scratch(self, enhanced: np.ndarray) -> List[_Candidate]:
        """划痕（class_id=0）：线状响应 + Canny 约束 + 细长几何筛选（§4.2 / §4.5）。"""
        cfg = self._cfg
        response = self._line_response(enhanced)
        _, binary = cv2.threshold(response, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges = cv2.Canny(enhanced, cfg.scratch_canny_low, cfg.scratch_canny_high)
        binary = cv2.bitwise_and(binary, cv2.dilate(edges, _K3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, _K3)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: List[_Candidate] = []
        for contour in contours:
            _, _, w, h = cv2.boundingRect(contour)
            short_side, long_side = min(int(w), int(h)), max(int(w), int(h))
            if short_side == 0 or long_side / short_side < cfg.scratch_min_aspect:
                continue  # 非细长形状，剔除块状误报
            if short_side > cfg.scratch_max_width:
                continue  # 超最大线宽，不是划痕
            mask = self._mask_of(contour, enhanced.shape)
            pixels = response[mask > 0]
            if pixels.size == 0:
                continue  # 无效候选：算不出分数就不输出（§4.2）
            candidates.append(_Candidate(0, contour, float(pixels.mean()) / 255.0, mask))
        return candidates

    def _detect_dent(self, gray: np.ndarray) -> List[_Candidate]:
        """凹坑（class_id=1）：LoG 响应 + Otsu + 圆形度/半径筛选（§4.2 / §4.5）。"""
        cfg = self._cfg
        blurred = cv2.GaussianBlur(gray, (0, 0), cfg.dent_log_sigma)
        log_abs = np.abs(cv2.Laplacian(blurred, cv2.CV_32F))
        # LoG 响应归一到 8bit 尺度，使 score=响应均值/255 落在 [0,1]（§4.2）
        response = cv2.normalize(log_abs, None, 0.0, 255.0, cv2.NORM_MINMAX)
        _, binary = cv2.threshold(
            response.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, _K3)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: List[_Candidate] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            perimeter = float(cv2.arcLength(contour, True))
            if area <= 0 or perimeter <= 0:
                continue
            radius = float(np.sqrt(area / np.pi))
            if not cfg.dent_radius_min <= radius <= cfg.dent_radius_max:
                continue
            roundness = 4.0 * np.pi * area / (perimeter * perimeter)
            if roundness < cfg.dent_roundness_min:
                continue  # 非点状形状，剔除细长杂讯
            mask = self._mask_of(contour, gray.shape)
            pixels = response[mask > 0]
            if pixels.size == 0:
                continue
            candidates.append(_Candidate(1, contour, float(pixels.mean()) / 255.0, mask))
        return candidates

    def _detect_stain(self, image: np.ndarray) -> List[_Candidate]:
        """污渍（class_id=2）：前景-背景 Lab 色距 + 饱和度阈值（§4.2 / §4.5）。"""
        cfg = self._cfg
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        background = np.median(lab.reshape(-1, 3), axis=0)  # 背景基准色
        dist = np.linalg.norm(lab - background, axis=2)  # 前景与背景的 Lab 欧氏色距

        mask = np.where(
            (dist >= cfg.stain_color_dist_min) & (hsv[:, :, 1] >= cfg.stain_sat_min), 255, 0
        ).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _K3)  # 去孤立噪点
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _K3)  # 连接同区域

        height, width = int(image.shape[0]), int(image.shape[1])
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: List[_Candidate] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area <= 0 or area / float(width * height) < cfg.stain_area_ratio_min:
                continue
            contour_mask = self._mask_of(contour, image.shape)
            pixels = dist[contour_mask > 0]
            if pixels.size == 0:
                continue
            candidates.append(
                _Candidate(2, contour, float(pixels.mean()) / LAB_DIST_MAX, contour_mask)
            )
        return candidates

    def _detect_notch(self, gray: np.ndarray) -> List[_Candidate]:
        """缺口（class_id=3）：边缘轮廓完整性 + 凸包缺陷深度（§4.2 / §4.5）。"""
        cfg = self._cfg
        _, part = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        part = cv2.morphologyEx(part, cv2.MORPH_CLOSE, _K3)
        contours, _ = cv2.findContours(part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) <= 0:
            return []
        # 轮廓在凸包顶点处出现重复坐标时（贴边 / 形态学缝合所致），
        # cv2.convexHull(returnPoints=False) 会返回非单调的索引序列，cv2.convexityDefects 直接报错。
        # 保序去重只去掉重复顶点：相邻重复（连续两点坐标相同，如 A-A-B）去重后面积/外接框不变；
        # 非相邻重复（"裂缝"型，如 A-B-A-C）去重会改变多边形面积，几何语义因此不等价。
        # 因此去重后的轮廓 **仅作凸包/缺陷的索引基准**：周长、面积、外接框仍按原轮廓计算；
        # 三角形 mask 取去重后轮廓上对应点位的坐标（与原轮廓同一坐标值，只是索引位移）。
        # 几何语义以原轮廓为准，避免影响既有帧的分数。
        indexed = self._dedupe_contour(contour)
        if len(indexed) < 3:
            return []
        hull = cv2.convexHull(indexed, returnPoints=False)
        if hull is None or len(hull) < 3:
            return []
        try:
            defects = cv2.convexityDefects(indexed, hull)
        except cv2.error as exc:  # 兜底：去重后仍病态的轮廓跳过，不整帧抛 DetectorInferenceError
            self._logger.debug(
                "缺口链路跳过异常轮廓：点数 %d，面积 %.1f，%s",
                len(contour),
                float(cv2.contourArea(contour)),
                exc,
            )
            return []
        if defects is None:
            return []
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0:
            return []
        diameter = perimeter / np.pi  # 等效直径作归一化分母，避免除零

        candidates: List[_Candidate] = []
        points = indexed.reshape(-1, 2)  # 缺陷索引以去重后的轮廓为基准，按坐标取点，避免索引错位
        for row in defects.reshape(-1, 4):
            start, end, far, depth_raw = int(row[0]), int(row[1]), int(row[2]), float(row[3])
            ratio = (depth_raw / 256.0) / diameter
            if ratio < cfg.notch_convexity_min:
                continue
            # 缺口区域 = 凸包边与缺陷点围成的三角形
            triangle = np.array(
                [points[start], points[end], points[far]], dtype=np.int32
            ).reshape(-1, 1, 2)
            mask = self._mask_of(triangle, gray.shape)
            if not mask.any():
                continue
            candidates.append(_Candidate(3, triangle, ratio, mask))
        return candidates

    @staticmethod
    def _dedupe_contour(contour: np.ndarray) -> np.ndarray:
        """保序去重轮廓顶点：保留首次出现的点，返回 int32 的 (N,1,2) 轮廓。"""
        points = contour.reshape(-1, 2).astype(np.int32, copy=False)
        _, first_index = np.unique(points, axis=0, return_index=True)
        return np.ascontiguousarray(points[np.sort(first_index)].reshape(-1, 1, 2))

    # ---------- 掩码（§4.3）----------

    @staticmethod
    def _mask_of(contour: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
        """生成 uint8、0/255、尺寸等于帧图像（H×W）的单通道掩码。"""
        mask = np.zeros(shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        return mask
