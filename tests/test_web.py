"""V1.2 扩展 03 验收：Web 演示服务（TestClient，不启动真实端口）。

覆盖：正常上传、坏文件、超大文件、错误扩展名、空文件、产物白名单、报告接口。
"""

from pathlib import Path

import cv2
import numpy as np
import yaml
from fastapi.testclient import TestClient

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
