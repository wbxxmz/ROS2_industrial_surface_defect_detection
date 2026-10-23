"""V1.2 扩展 03：Web 路由（检测 / 产物读取 / 单样本报告）。

并发模型（补充约束）：
- ``app.state.detector`` 单例；``load`` 用 ``threading.Lock`` 双重检查，防止首个并发请求重复加载；
- 每次 ``detect``（含分级 / 判定 / 渲染）在同一把锁内串行；
- 端点均为 ``async def``，同步推理通过 ``asyncio.to_thread`` 下放，不阻塞事件循环。

异常映射：``SourceOpenError`` → 422、``DetectorInferenceError`` → 500、
``ValueError`` → 400、其余未预期异常 → 500 且只记服务端日志（见 ``web/app.py``）。
"""

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from judge import judge
from pipeline.pipeline import build_detector
from report import ReportBuilder
from severity import grade
from utils.exceptions import DetectorInferenceError
from visualization import render_to_file
from visualization.attention import generate_eigencam_heatmap

logger = logging.getLogger("defect.web")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "web"

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 单文件 ≤ 10 MB
REPORT_FORMATS = {"html": "text/html", "pdf": "application/pdf"}

router = APIRouter()


def _error(status: int, error: str, detail: str) -> JSONResponse:
    """统一错误响应体 ``{"error": "...", "detail": "..."}``。"""
    return JSONResponse(status_code=status, content={"error": error, "detail": detail})


def _ensure_detector(app) -> object:
    """单例懒加载：双重检查，首个并发请求只加载一次。"""
    detector = app.state.detector
    if detector is not None:
        return detector
    with app.state.lock:
        if app.state.detector is None:
            loaded = build_detector(app.state.config)
            loaded.load(app.state.config.get("detector") or {})
            app.state.detector = loaded
            logger.info("检测器已加载：%s %s", loaded.name, loaded.version)
    return app.state.detector


def _detections_payload(detections) -> list:
    """Detection → JSON（bbox 元组转 list，不携带 mask）。"""
    return [
        {
            "class_id": det.class_id,
            "class_name": det.class_name,
            "bbox": list(det.bbox),
            "area_px": det.area_px,
            "area_ratio": det.area_ratio,
            "confidence": det.confidence,
            "severity": det.severity,
        }
        for det in detections
    ]


def _run_detection(app, filename: str, image: np.ndarray) -> dict:
    """同步执行：检测 → 分级 → 判定 → 标注图（→ 注意力热力图），整体在锁内串行。"""
    from core import Frame

    detector = _ensure_detector(app)
    frame = Frame(
        image=image,
        width=image.shape[1],
        height=image.shape[0],
        frame_index=0,
        timestamp=0.0,
    )
    config = app.state.config
    report_cfg = config.get("report") or {}
    request_id = uuid.uuid4().hex[:12]
    output_dir = Path(app.state.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with app.state.lock:
        detections = grade(detector.detect(frame), config.get("severity"))
        verdict = judge(detections, config.get("judge"))

        # 文件名与报告模块约定一致：attention_<stem>.jpg 与标注图同目录，报告自动关联
        annotated_path = output_dir / f"{request_id}.jpg"
        render_to_file(frame, detections, verdict, str(annotated_path))
        artifacts = app.state.artifacts
        artifacts[annotated_path.name] = str(annotated_path)

        attention_path: Optional[Path] = None
        if bool(report_cfg.get("attention", False)) and getattr(detector, "model", None) is not None:
            try:
                attention_path = Path(
                    generate_eigencam_heatmap(
                        frame, detector, output_dir / f"attention_{request_id}.jpg"
                    )
                )
                artifacts[attention_path.name] = str(attention_path)
            except (DetectorInferenceError, OSError, ValueError) as exc:
                logger.warning("注意力热力图生成失败（不影响检测结果）：%s", exc)
                attention_path = None

        source = Path(filename).stem or "upload"
        app.state.current_sample = {
            "id": request_id,
            "batch": {
                "source": source,
                "verdict": verdict,
                "detections": detections,
                "image_path": str(annotated_path),
            },
        }

    return {
        "sample_id": request_id,
        "source": source,
        "is_pass": verdict.is_pass,
        "result": "PASS" if verdict.is_pass else "FAIL",
        "reason": verdict.reason,
        "total_count": verdict.total_count,
        "max_severity": verdict.max_severity,
        "detections": _detections_payload(detections),
        "image_size": {"width": frame.width, "height": frame.height},
        "annotated_url": f"/api/images/{annotated_path.name}",
        "attention_url": f"/api/images/{attention_path.name}" if attention_path else None,
        "report_urls": {
            "html": f"/api/report?format=html&sample_id={request_id}",
            "pdf": f"/api/report?format=pdf&sample_id={request_id}",
        },
    }


async def _decode_upload(file: UploadFile):
    """校验扩展名 / 大小 / 可解码性；返回 ``(filename, image)`` 或错误响应。"""
    filename = Path(file.filename or "").name
    if not filename or Path(filename).suffix.lower() not in ALLOWED_SUFFIXES:
        return None, _error(422, "InvalidFileType", "仅支持 .jpg / .jpeg / .png / .bmp 图片")

    contents = await file.read()
    if not contents:
        return None, _error(422, "EmptyFile", "上传文件为空")
    if len(contents) > MAX_UPLOAD_BYTES:
        return None, _error(413, "FileTooLarge", "单文件不得超过 10 MB")

    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return None, _error(422, "DecodeError", "图片解码失败，请确认文件未损坏")
    return (filename, image), None


@router.get("/", include_in_schema=False)
async def index():
    """返回拖拽上传页。"""
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


@router.post("/api/detect")
async def detect(request: Request, file: UploadFile = File(...)):
    """上传单张图片，返回判定结论、缺陷明细与产物地址。"""
    decoded, error = await _decode_upload(file)
    if error is not None:
        return error
    filename, image = decoded
    return await asyncio.to_thread(_run_detection, request.app, filename, image)


@router.get("/api/images/{name}")
async def image(name: str, request: Request):
    """只返回本服务生成的产物；文件名白名单校验，拒绝路径穿越。"""
    if Path(name).name != name or name not in request.app.state.artifacts:
        return _error(404, "NotFound", "该文件不是本服务生成的产物")
    return FileResponse(request.app.state.artifacts[name], media_type="image/jpeg")


def _build_report(app, sample: dict, fmt: str) -> Path:
    """仅对当前样本生成报告（batch=[当前样本]，不做跨请求聚合）。"""
    cfg = dict(app.state.config.get("report") or {})
    output = Path(app.state.output_dir) / f"{sample['id']}_report.{fmt}"
    cfg.update({"format": fmt, "output_path": str(output)})
    return ReportBuilder().build([sample["batch"]], cfg)


@router.get("/api/report")
async def report(
    request: Request,
    fmt: str = Query("html", alias="format"),
    sample_id: Optional[str] = None,
):
    """为当前样本生成并返回报告文件。"""
    fmt = str(fmt).lower()
    if fmt not in REPORT_FORMATS:
        return _error(400, "InvalidFormat", "format 仅支持 html / pdf")

    current = request.app.state.current_sample
    if current is None:
        return _error(409, "NoSample", "当前没有可生成报告的样本，请先上传图片")
    if sample_id is not None and sample_id != current["id"]:
        return _error(409, "StaleSample", "只能为当前样本生成报告")

    path = await asyncio.to_thread(_build_report, request.app, current, fmt)
    return FileResponse(
        path,
        media_type=REPORT_FORMATS[fmt],
        filename=f"quality_report_{current['id']}.{fmt}",
    )
