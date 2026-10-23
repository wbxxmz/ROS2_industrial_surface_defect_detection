"""V1.2 扩展 03 验收：Web 演示服务（TestClient，不启动真实端口）。

覆盖：正常上传、坏文件、超大文件、错误扩展名、空文件、产物白名单、报告接口。
"""

import threading
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from fastapi.testclient import TestClient

from core.types import Detection
from severity.grader import grade
from utils.exceptions import DetectorInferenceError
from web.app import create_app

MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def write_config(tmp_path: Path) -> Path:
    """最小配置：dummy 检测器（零模型依赖）+ 临时产物目录。"""
    config = {
        "input": {"type": "image", "dir": "."},
        "detector": {"type": "dummy", "params": {}},
        "judge": {},
        "severity": {},
        "web": {"output_dir": str(tmp_path / "web_output")},
        "report": {
            "enabled": True,
            "format": "html",
            "output_path": str(tmp_path / "report.html"),
            "title": "Web 测试报告",
            "algorithm_name": "dummy",
            "algorithm_version": "0.1.0",
            "sample_source": "test",
            "attention": False,
        },
    }
    path = tmp_path / "web_test.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return path


def client_for(tmp_path: Path) -> TestClient:
    return TestClient(create_app(write_config(tmp_path)))


def jpeg_bytes() -> bytes:
    image = np.full((48, 64, 3), 128, dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    return buffer.tobytes()


def write_fail_config(tmp_path: Path) -> Path:
    """在最小配置基础上启用严重度分级，用于覆盖 FAIL 分支（不改现有 write_config）。"""
    path = write_config(tmp_path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["severity"] = {"enabled": True}
    config["report"]["attention"] = False
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return path


class _StubDetector:
    """最小检测器桩：满足 name/version/load/detect/release，刻意不提供 model 属性。"""

    name = "stub"
    version = "0.0.0"

    def load(self, _cfg=None):
        return None

    def detect(self, _frame):
        raise NotImplementedError

    def release(self):
        return None


class _SingleDefectDetector(_StubDetector):
    """恒定返回 1 条 stain 缺陷，用于验证 FAIL 分支。"""

    def detect(self, frame):
        return [
            Detection(
                class_id=2,
                class_name="stain",
                bbox=(20, 10, 16, 16),
                area_px=200,
                area_ratio=0.065,
                confidence=0.9,
                mask=None,
                severity=None,
            )
        ]


class _RaisingDetector(_StubDetector):
    """detect() 抛指定异常，用于验证全局异常映射。"""

    def __init__(self, error: Exception):
        self._error = error

    def detect(self, frame):
        raise self._error


def test_detect_success_and_image(tmp_path):
    """正常上传：返回判定 + 产物地址，标注图可读取。"""
    with client_for(tmp_path) as client:
        response = client.post(
            "/api/detect", files={"file": ("sample.jpg", jpeg_bytes(), "image/jpeg")}
        )
        assert response.status_code == 200
        payload = response.json()

        assert payload["is_pass"] is True and payload["result"] == "PASS"
        assert payload["detections"] == []  # dummy 检测器恒无缺陷
        assert payload["annotated_url"].startswith("/api/images/")
        assert payload["attention_url"] is None
        assert payload["image_size"] == {"width": 64, "height": 48}

        image = client.get(payload["annotated_url"])
        assert image.status_code == 200
        assert image.headers["content-type"] == "image/jpeg"


def test_detect_bad_file(tmp_path):
    """扩展名合法但内容不可解码：422 + 统一错误体。"""
    with client_for(tmp_path) as client:
        response = client.post(
            "/api/detect", files={"file": ("bad.jpg", b"not-an-image", "image/jpeg")}
        )
        assert response.status_code == 422
        body = response.json()
        assert set(body) == {"error", "detail"}
        assert body["error"] == "DecodeError"


def test_detect_oversize(tmp_path):
    """超过 10 MB：413，不进入解码。"""
    with client_for(tmp_path) as client:
        huge = b"0" * (MAX_UPLOAD_BYTES + 1)
        response = client.post("/api/detect", files={"file": ("huge.jpg", huge, "image/jpeg")})
        assert response.status_code == 413
        assert response.json()["error"] == "FileTooLarge"


def test_detect_wrong_extension(tmp_path):
    """非白名单扩展名：422，即使内容是合法图片。"""
    with client_for(tmp_path) as client:
        response = client.post(
            "/api/detect", files={"file": ("sample.gif", jpeg_bytes(), "image/gif")}
        )
        assert response.status_code == 422
        assert response.json()["error"] == "InvalidFileType"


def test_detect_empty_file(tmp_path):
    """空文件：422。"""
    with client_for(tmp_path) as client:
        response = client.post("/api/detect", files={"file": ("empty.jpg", b"", "image/jpeg")})
        assert response.status_code == 422
        assert response.json()["error"] == "EmptyFile"


def test_images_reject_unknown_names(tmp_path):
    """只返回本服务生成的产物，拒绝路径穿越 / 任意文件名。"""
    with client_for(tmp_path) as client:
        assert client.get("/api/images/not_generated.jpg").status_code == 404
        assert client.get("/api/images/..%2Fapp.py").status_code == 404


def test_report_requires_current_sample(tmp_path):
    """未上传图片时请求报告：409。"""
    with client_for(tmp_path) as client:
        response = client.get("/api/report?format=html")
        assert response.status_code == 409
        assert response.json()["error"] == "NoSample"


def test_report_only_for_current_sample(tmp_path):
    """报告只服务当前样本：正常下载、旧 sample_id 拒绝、非法格式 400。"""
    with client_for(tmp_path) as client:
        detect = client.post(
            "/api/detect", files={"file": ("sample.jpg", jpeg_bytes(), "image/jpeg")}
        ).json()

        report = client.get("/api/report?format=html")
        assert report.status_code == 200
        assert report.headers["content-type"].startswith("text/html")
        assert "Web 测试报告" in report.text

        stale = client.get("/api/report?format=html&sample_id=deadbeef")
        assert stale.status_code == 409 and stale.json()["error"] == "StaleSample"

        invalid = client.get("/api/report?format=xlsx")
        assert invalid.status_code == 400 and invalid.json()["error"] == "InvalidFormat"

        fresh = client.get(detect["report_urls"]["html"])
        assert fresh.status_code == 200


def test_detect_fail_branch(tmp_path, monkeypatch):
    """FAIL 分支：单条 stain 触发判定失败，severity 以 grade() 重算结果为准。"""
    monkeypatch.setattr("web.routes.build_detector", lambda cfg: _SingleDefectDetector())
    app = create_app(write_fail_config(tmp_path))

    expected = grade(
        [
            Detection(
                class_id=2,
                class_name="stain",
                bbox=(20, 10, 16, 16),
                area_px=200,
                area_ratio=0.065,
                confidence=0.9,
                mask=None,
                severity=None,
            )
        ],
        {"enabled": True},
    )[0].severity

    with TestClient(app) as client:
        response = client.post(
            "/api/detect", files={"file": ("stain.jpg", jpeg_bytes(), "image/jpeg")}
        )
        assert response.status_code == 200
        payload = response.json()

        assert payload["is_pass"] is False
        assert payload["result"] == "FAIL"
        assert payload["total_count"] == 1
        assert len(payload["detections"]) == 1
        assert payload["max_severity"] == expected

        detection = payload["detections"][0]
        assert detection["class_name"] == "stain"
        assert detection["confidence"] == 0.9
        assert detection["bbox"] == [20, 10, 16, 16]
        assert detection["severity"] == expected

        image = client.get(payload["annotated_url"])
        assert image.status_code == 200
        assert image.headers["content-type"].startswith("image/")


def test_detector_inference_error_maps_500(tmp_path, monkeypatch):
    """detect() 抛 DetectorInferenceError：500 + 统一错误体，detail 原样且无堆栈。"""
    monkeypatch.setattr(
        "web.routes.build_detector",
        lambda cfg: _RaisingDetector(DetectorInferenceError("模拟推理失败")),
    )

    with TestClient(create_app(write_config(tmp_path))) as client:
        response = client.post(
            "/api/detect", files={"file": ("sample.jpg", jpeg_bytes(), "image/jpeg")}
        )

    assert response.status_code == 500
    body = response.json()
    assert set(body) == {"error", "detail"}
    assert body["error"] == "DetectorInferenceError"
    assert body["detail"] == "模拟推理失败"
    assert "traceback" not in body["detail"].lower()


def test_unexpected_error_maps_500(tmp_path, monkeypatch):
    """detect() 抛 RuntimeError：500 + 固定文案，不透传原始异常。"""
    monkeypatch.setattr(
        "web.routes.build_detector", lambda cfg: _RaisingDetector(RuntimeError("boom"))
    )
    app = create_app(write_config(tmp_path))

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/detect", files={"file": ("sample.jpg", jpeg_bytes(), "image/jpeg")}
        )

    assert response.status_code == 500
    body = response.json()
    assert set(body) == {"error", "detail"}
    assert body["error"] == "InternalError"
    assert body["detail"] == "服务内部错误"
    assert "traceback" not in body["detail"].lower()
    assert "boom" not in body


def test_detector_load_once_under_concurrency(tmp_path, monkeypatch):
    """双线程同时发首次请求：build_detector 只被调用一次。

    注意：本用例对「锁内二次检查」是概率性覆盖——依赖两个请求在 load 的
    0.25s 睡眠窗口内同时到达 fast path，不保证 100% 触发竞争；断言只校验
    「只构建一次」这一结果不变量。
    """
    calls = {"n": 0}

    class _SlowDetector(_StubDetector):
        def load(self, _cfg=None):
            time.sleep(0.25)

        def detect(self, frame):
            return []

    def factory(cfg):
        calls["n"] += 1
        return _SlowDetector()

    monkeypatch.setattr("web.routes.build_detector", factory)
    app = create_app(write_config(tmp_path))

    barrier = threading.Barrier(2)
    statuses = []
    errors = []

    with TestClient(app) as client:

        def worker():
            try:
                barrier.wait(timeout=5)
                response = client.post(
                    "/api/detect", files={"file": ("concurrent.jpg", jpeg_bytes(), "image/jpeg")}
                )
                statuses.append(response.status_code)
            except Exception as exc:  # 线程内异常带回主线程断言
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        assert not any(thread.is_alive() for thread in threads)

        # 必须在 lifespan 退出前断言：shutdown 会 release() 并把 detector 置回 None
        assert app.state.detector is not None
        assert calls["n"] == 1

    assert errors == []
    assert statuses == [200, 200]
