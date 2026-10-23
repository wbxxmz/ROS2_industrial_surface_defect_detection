"""V1.2 扩展 03：FastAPI 应用工厂（模型单例懒加载 + 显式释放）。

- ``create_app(config_path)`` 为唯一入口，``uvicorn web.app:create_app --factory`` 可直接启动；
- 检测器不在启动时加载：首个请求到达才 ``load``，避免 CLI 路线被 Web 依赖拖慢；
- 应用关闭时在 lifespan 中 ``release()``，释放显存缓存（幂等）。
"""

import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from utils.config import load_config
from utils.exceptions import DetectorInferenceError, ProjectError, SourceOpenError
from web.routes import DEFAULT_OUTPUT_DIR, router

logger = logging.getLogger("defect.web")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "html_yolo_demo.yaml"


def _error(status: int, error: str, detail: str) -> JSONResponse:
    """统一错误响应体：``{"error": "...", "detail": "..."}``，不返回 traceback。"""
    return JSONResponse(status_code=status, content={"error": error, "detail": detail})


def _register_error_handlers(app: FastAPI) -> None:
    """异常 → HTTP 状态码映射（扩展 03 补充约束）。"""

    @app.exception_handler(SourceOpenError)
    async def _handle_source(_request, exc: SourceOpenError):
        logger.warning("输入源异常：%s", exc)
        return _error(422, "SourceOpenError", str(exc))

    @app.exception_handler(DetectorInferenceError)
    async def _handle_inference(_request, exc: DetectorInferenceError):
        logger.error("推理异常：%s", exc)
        return _error(500, "DetectorInferenceError", str(exc))

    @app.exception_handler(ValueError)
    async def _handle_value(_request, exc: ValueError):
        logger.warning("参数异常：%s", exc)
        return _error(400, "ValueError", str(exc))

    @app.exception_handler(RequestValidationError)
    async def _handle_validation(_request, _exc: RequestValidationError):
        return _error(422, "ValidationError", "请求参数不合法")

    @app.exception_handler(ProjectError)
    async def _handle_project(_request, exc: ProjectError):
        logger.error("服务异常：%s", exc)
        return _error(500, type(exc).__name__, str(exc))

    @app.exception_handler(Exception)
    async def _handle_unexpected(_request, exc: Exception):
        # 只记服务端日志，响应体不透出堆栈
        logger.exception("未预期异常：%s", exc)
        return _error(500, "InternalError", "服务内部错误")


def create_app(config_path: str | Path | None = None) -> FastAPI:
    """创建应用实例。

    :param config_path: 配置文件路径；缺省读环境变量 ``DEFECT_WEB_CONFIG``，
        再缺省用 ``configs/html_yolo_demo.yaml``
    """
    path = Path(config_path or os.environ.get("DEFECT_WEB_CONFIG") or DEFAULT_CONFIG_PATH)
    config = load_config(path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        with app.state.lock:
            if app.state.detector is not None:
                app.state.detector.release()
                app.state.detector = None
                logger.info("已释放检测器与显存缓存")

    app = FastAPI(title="工业表面缺陷检测 Web 演示", version="1.2.0", lifespan=lifespan)
    app.state.config = config
    app.state.config_path = str(path)
    app.state.detector = None  # 懒加载单例
    app.state.lock = threading.Lock()  # 加载 + 每次 detect 串行化
    app.state.artifacts = {}  # 文件名 → 绝对路径，仅登记本服务本次生成的产物
    app.state.current_sample = None  # 最近一次检测结果（报告仅服务当前样本）
    app.state.output_dir = Path((config.get("web") or {}).get("output_dir") or DEFAULT_OUTPUT_DIR)

    app.include_router(router)
    _register_error_handlers(app)
    return app
