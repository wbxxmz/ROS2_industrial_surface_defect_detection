"""阶段 4 §7 验收 #5：YoloDetector 单元测试（契约字段 / 异常语义 / 生命周期 / 配置校验 / 映射表一致）。

需要真实 torch + ultralytics 的用例用 ``pytest.importorskip`` 守卫；
其余用例通过注入假模型（stub）验证转换与生命周期逻辑，不依赖推理引擎。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import Detection, Frame  # noqa: E402
from detectors.opencv_detector import CLASS_MAP  # noqa: E402
from detectors.yolo_detector import YoloDetector, YoloDetectorConfig  # noqa: E402
from utils.exceptions import DetectorInferenceError, DetectorLoadError  # noqa: E402

FRAME_W, FRAME_H = 200, 200
IMAGE_AREA = FRAME_W * FRAME_H


# ---------- 素材构造 ----------


def _frame(width: int = FRAME_W, height: int = FRAME_H) -> Frame:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    return Frame(image=image, width=width, height=height, frame_index=0, timestamp=0.0)


def _model_file(tmp_path: Path, name: str = "fake.pt") -> str:
    """合法存在的权重占位文件（仅用于通过路径校验）。"""
    path = tmp_path / name
    path.write_bytes(b"not-a-real-model")
    return str(path)


class _FakeBoxes:
    """模拟 ultralytics ``results.boxes``（numpy，无 torch 依赖）。"""

    def __init__(self, rows) -> None:
        data = np.asarray(rows, dtype=np.float32).reshape(-1, 6)
        self.xyxy = data[:, :4]
        self.conf = data[:, 4]
        self.cls = data[:, 5]

    def __len__(self) -> int:
        return int(self.xyxy.shape[0])


class _FakeResult:
    def __init__(self, rows) -> None:
        self.boxes = _FakeBoxes(rows) if rows else None


class _FakeModel:
    """假模型：predict 返回预置结果，可用于验证转换逻辑。"""

    def __init__(self, results) -> None:
        self._results = results
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self._results, Exception):
            raise self._results
        return self._results

    def eval(self):
        return self

    def to(self, device):
        return self


def _loaded_detector(model, tmp_path: Path, **params) -> YoloDetector:
    """注入假模型，跳过真实权重加载但保留 §4.4 参数校验路径。"""
    detector = YoloDetector()
    base = {"model_path": _model_file(tmp_path, "stub.pt")}
    base.update(params)
    detector._cfg = YoloDetectorConfig.from_dict(base)
    detector._model = model
    return detector


# ---------- §4.4 配置校验 ----------


def test_config_defaults_and_valid_load(tmp_path):
    """合法参数返回 None；未给出的字段用 §4.4 表内默认值。"""
    path = _model_file(tmp_path)
    cfg = YoloDetectorConfig.from_dict({"model_path": path})
    assert cfg.conf_threshold == 0.25
    assert cfg.iou_threshold == 0.45
    assert cfg.device == "cpu"
    assert cfg.imgsz == 640


@pytest.mark.parametrize(
    "params",
    [
        {},  # 缺必填 model_path
        {"model_path": "no_such_model.pt"},  # 文件不存在
        {"model_path": ""},  # 空路径
        {"model_path": "X", "conf_threshold": 1.5},  # 置信度越界
        {"model_path": "X", "conf_threshold": -0.1},
        {"model_path": "X", "iou_threshold": 2.0},  # IoU 越界
        {"model_path": "X", "imgsz": 0},  # 非正整数
        {"model_path": "X", "device": "gpu"},  # 非法设备名
        {"model_path": "X", "device": "cuda:abc"},  # 非法设备号
        {"model_path": "X", "unknown_field": 1},  # 未知字段
    ],
)
def test_config_rejects_invalid_params(params):
    with pytest.raises(DetectorLoadError):
        YoloDetectorConfig.from_dict(params)


def test_load_requires_params_key():
    with pytest.raises(DetectorLoadError):
        YoloDetector().load({"type": "yolo"})


def test_load_missing_model_file_raises_detector_load_error():
    with pytest.raises(DetectorLoadError):
        YoloDetector().load({"params": {"model_path": "definitely_missing.pt"}})


# ---------- §5.1 异常语义 ----------


def test_detect_none_frame_raises_value_error():
    with pytest.raises(ValueError):
        YoloDetector().detect(None)


@pytest.mark.parametrize("shape", [(0, 0, 3), (0, FRAME_W, 3), (FRAME_H, 0, 3)])
def test_detect_zero_size_frame_raises_value_error(shape):
    image = np.zeros(shape, dtype=np.uint8)
    frame = Frame(image=image, width=0, height=0, frame_index=0, timestamp=0.0)
    with pytest.raises(ValueError):
        YoloDetector().detect(frame)


def test_detect_size_mismatch_raises_value_error():
    frame = Frame(
        image=np.zeros((10, 10, 3), dtype=np.uint8), width=99, height=10, frame_index=0, timestamp=0.0
    )
    with pytest.raises(ValueError):
        YoloDetector().detect(frame)


def test_detect_without_load_raises_inference_error():
    with pytest.raises(DetectorInferenceError):
        YoloDetector().detect(_frame())


def test_detect_converts_inference_failure_to_inference_error(tmp_path):
    """推理异常（含 CUDA OOM / RuntimeError）统一转译为 DetectorInferenceError 并保留原因。"""
    model = _FakeModel(RuntimeError("CUDA out of memory"))
    detector = _loaded_detector(model, tmp_path)
    with pytest.raises(DetectorInferenceError) as excinfo:
        detector.detect(_frame())
    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_detect_empty_result_returns_empty_list(tmp_path):
    detector = _loaded_detector(_FakeModel([_FakeResult([])]), tmp_path)
    assert detector.detect(_frame()) == []


# ---------- §4.2 字段转换 ----------


def test_detection_fields_follow_contract(tmp_path):
    """[x1,y1,x2,y2,conf,cls] → Detection：像素 bbox、area_px=w*h、confidence∈[0,1]。"""
    rows = [[10, 20, 60, 80, 0.9, 1], [100, 100, 140, 130, 0.5, 2]]
    detector = _loaded_detector(_FakeModel([_FakeResult(rows)]), tmp_path)

    detections = detector.detect(_frame())

    assert isinstance(detections, list) and len(detections) == 2
    first, second = detections
    assert isinstance(first, Detection)
    assert first.class_id == 1 and first.class_name == CLASS_MAP[1] == "dent"
    assert first.bbox == (10, 20, 50, 60)  # xyxy → (x, y, w, h)，像素 int
    assert first.area_px == 50 * 60
    assert first.area_ratio == pytest.approx(first.area_px / IMAGE_AREA)
    assert first.confidence == pytest.approx(0.9)
    assert first.mask is None and first.severity is None

    assert second.class_id == 2 and second.class_name == "stain"
    assert second.bbox == (100, 100, 40, 30)
    assert 0.0 <= second.confidence <= 1.0


def test_class_id_outside_mapping_is_dropped(tmp_path):
    """class_id 不在 0~3 内的结果丢弃，不影响同帧其它合法结果。"""
    rows = [[0, 0, 10, 10, 0.9, 7], [10, 10, 50, 50, 0.8, 0]]
    detector = _loaded_detector(_FakeModel([_FakeResult(rows)]), tmp_path)

    detections = detector.detect(_frame())

    assert len(detections) == 1
    assert detections[0].class_id == 0 and detections[0].class_name == "scratch"


def test_bbox_is_clamped_to_image_bounds(tmp_path):
    """越界框 clamp 到 [0,W]×[0,H]；clamp 后退化的框丢弃。"""
    rows = [[-5, -5, 30, 40, 0.9, 0], [190, 190, 260, 260, 0.9, 0]]
    detector = _loaded_detector(_FakeModel([_FakeResult(rows)]), tmp_path)

    detections = detector.detect(_frame())

    assert len(detections) == 2
    for detection in detections:
        x, y, w, h = detection.bbox
        assert x >= 0 and y >= 0 and w >= 1 and h >= 1
        assert x + w <= FRAME_W and y + h <= FRAME_H
    assert detections[0].bbox == (0, 0, 30, 40)
    assert detections[1].bbox == (190, 190, 10, 10)


def test_predict_receives_configured_thresholds(tmp_path):
    """conf/iou/imgsz/device 全部来自 load(cfg)，未写死。"""
    model = _FakeModel([_FakeResult([])])
    detector = _loaded_detector(
        model, tmp_path, conf_threshold=0.66, iou_threshold=0.33, imgsz=320, device="cpu"
    )
    detector.detect(_frame())
    call = model.calls[0]
    assert call["conf"] == pytest.approx(0.66)
    assert call["iou"] == pytest.approx(0.33)
    assert call["imgsz"] == 320
    assert call["device"] == "cpu"


# ---------- §5.2 生命周期 ----------


def test_release_is_idempotent(tmp_path):
    detector = _loaded_detector(_FakeModel([_FakeResult([])]), tmp_path)
    assert detector.release() is None
    assert detector._model is None
    assert detector.release() is None  # 重复调用不抛异常


def test_detect_after_release_raises_inference_error(tmp_path):
    detector = _loaded_detector(_FakeModel([_FakeResult([])]), tmp_path)
    detector.release()
    with pytest.raises(DetectorInferenceError):
        detector.detect(_frame())


def test_cuda_unavailable_raises_detector_load_error(tmp_path):
    """device 指定 cuda 但环境不可用 → DetectorLoadError，不静默回退。"""
    torch = pytest.importorskip("torch")
    pytest.importorskip("ultralytics")
    if torch.cuda.is_available():
        pytest.skip("本机 CUDA 可用，无法验证不可用分支")
    with pytest.raises(DetectorLoadError):
        YoloDetector().load({"params": {"model_path": _model_file(tmp_path), "device": "cuda:0"}})


def test_name_and_version_contract(tmp_path):
    detector = YoloDetector()
    assert detector.name == "yolo" and detector.version == "0.1.0"
    loaded = _loaded_detector(_FakeModel([]), tmp_path)
    assert loaded.name == "stub"  # 权重文件 stem，用于报告溯源


# ---------- 映射表一致性 ----------


def test_class_map_reused_from_stage3():
    assert CLASS_MAP == {0: "scratch", 1: "dent", 2: "stain", 3: "notch"}


# ---------- 真实推理（依赖可用时才跑）----------


def test_real_yolov8n_inference_smoke(tmp_path):
    """真实 ultralytics 推理：权重不存在时由 load 抛 DetectorLoadError 而非崩溃。"""
    pytest.importorskip("torch")
    pytest.importorskip("ultralytics")
    with pytest.raises(DetectorLoadError):
        YoloDetector().load({"params": {"model_path": str(tmp_path / "missing.pt")}})
