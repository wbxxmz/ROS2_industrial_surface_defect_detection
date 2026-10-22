"""V1.1 扩展 01 验收：HTML 交互式报告单元测试。

覆盖：正常生成（单文件 + base64 内嵌 + 无外链）、特殊字符转义、图片缺失降级。
"""

from pathlib import Path

import cv2
import numpy as np

from core.types import Detection
from core.verdict import Verdict
from report import ReportBuilder


def make_detection(class_id=0, severity=1, confidence=0.5):
    """构造一条符合阶段 0 契约的 Detection。"""
    return Detection(
        class_id=class_id,
        class_name={0: "scratch", 1: "dent", 2: "stain", 3: "notch"}[class_id],
        bbox=(10, 20, 30, 40),
        area_px=1200,
        area_ratio=0.03,
        confidence=confidence,
        mask=None,
        severity=severity,
    )


def make_verdict(detections=None, reason="检测到缺陷"):
    """构造与 detections 口径一致的 Verdict。"""
    detections = detections or []
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for det in detections:
        class_counts[det.class_id] += 1
    return Verdict(
        is_pass=False,
        reason=reason,
        total_count=len(detections),
        class_counts=class_counts,
        total_area_px=sum(det.area_px for det in detections),
        total_area_ratio=min(1.0, sum(det.area_ratio for det in detections)),
        triggered=["count_exceeded"],
        max_severity=1,
    )


def make_batch(image_path=None, reason="检测到缺陷", count=1):
    """构造批量记录：每份样本 1 条 scratch 缺陷。"""
    batch = []
    for index in range(count):
        detections = [make_detection(confidence=0.5 + index * 0.1)]
        batch.append(
            {
                "source": f"sample_{index:03d}",
                "verdict": make_verdict(detections=detections, reason=reason),
                "detections": detections,
                "image_path": image_path,
            }
        )
    return batch


def html_cfg(tmp_path):
    """HTML 格式的报告配置。"""
    return {
        "output_path": str(tmp_path / "quality_report.html"),
        "format": "html",
        "title": "测试报告",
        "algorithm_name": "OpenCV",
        "algorithm_version": "0.1.0",
        "params": {"min_area": 10},
    }


def write_sample_image(path: Path) -> Path:
    """写一张可被 base64 内嵌的示例图。"""
    image = np.full((64, 64, 3), 128, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)
    return path


def test_html_report_generated(tmp_path):
    """正常生成：单文件落盘、含内嵌图片、无外链资源。"""
    image_path = write_sample_image(tmp_path / "visual.jpg")
    output = ReportBuilder().build(make_batch(image_path=image_path), html_cfg(tmp_path))

    assert output.exists() and output.suffix == ".html"
    text = output.read_text(encoding="utf-8")

    assert "<!DOCTYPE html>" in text
    assert "data:image/jpeg;base64," in text  # 标注图内嵌
    assert "data:image/png;base64," in text  # 统计图内嵌
    assert 'src="http://' not in text and 'src="https://' not in text  # 无外链
    assert "缺陷明细" in text and "样本抽检图" in text
    assert "sample_000" in text
    # 临时统计图不残留
    assert not (tmp_path / "quality_report_class_chart.png").exists()


def test_html_escapes_special_characters(tmp_path):
    """含 > < & 的 reason 原样显示且不破坏页面结构。"""
    reason = "<script>alert('x&y')</script> a > b"
    output = ReportBuilder().build(make_batch(reason=reason), html_cfg(tmp_path))
    text = output.read_text(encoding="utf-8")

    assert "<script>alert" not in text  # 未被当作标签执行
    assert "&lt;script&gt;" in text
    assert "a &gt; b" in text
    assert "x&amp;y" in text


def test_html_missing_image_placeholder(tmp_path):
    """image_path 指向的图片缺失：灰底占位 +「图片缺失」，不抛异常。"""
    missing = tmp_path / "not_exists.jpg"
    output = ReportBuilder().build(make_batch(image_path=missing), html_cfg(tmp_path))
    text = output.read_text(encoding="utf-8")

    assert "图片缺失" in text
    assert 'class="shot missing"' in text
    # 统计图本身正常内嵌，缺失的只是标注图
    assert "data:image/png;base64," in text


def test_html_is_supported_format(tmp_path):
    """html 已纳入 SUPPORTED_FORMATS，不再抛「不支持格式」。"""
    output = ReportBuilder().build(make_batch(), html_cfg(tmp_path))
    assert output.exists()
