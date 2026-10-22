"""V1.1 扩展验收：双算法对比 HTML（数据对齐 + 渲染）。"""

from pathlib import Path

import cv2
import numpy as np

from report.exporters.comparison_html import export_comparison_html
from scripts.compare_html import align_samples, build_summary


def record(name, is_pass=True, count=1, severity=0, time_ms=10.0, image_path=None):
    """构造单算法侧的单样本记录。"""
    return {
        "name": name,
        "image_path": image_path,
        "is_pass": is_pass,
        "total_count": count,
        "max_severity": severity,
        "time_ms": time_ms,
    }


def write_image(path: Path) -> Path:
    assert cv2.imwrite(str(path), np.full((32, 32, 3), 100, dtype=np.uint8))
    return path


def sample_data(tmp_path):
    """构造一组对比数据（两张样本：一致 / 不一致）。"""
    image = write_image(tmp_path / "visual.jpg")
    samples = [
        {
            "name": "crazing_98",
            "consistent": False,
            "opencv": record("crazing_98", is_pass=True, count=0, severity=None, time_ms=5.5, image_path=str(image)),
            "yolo": record("crazing_98", is_pass=False, count=5, severity=2, time_ms=17.5, image_path=str(image)),
        },
        {
            "name": "crazing_148",
            "consistent": True,
            "opencv": record("crazing_148", is_pass=False, count=2, severity=1, time_ms=6.0, image_path=str(image)),
            "yolo": record("crazing_148", is_pass=False, count=4, severity=1, time_ms=18.0, image_path=str(image)),
        },
    ]
    return {"generated_at": "2026-10-22 20:00:00", "input_dir": "test_picture", "samples": samples,
            "summary": build_summary(samples)}


def test_align_samples_by_stem():
    """按文件名主干对齐：单侧缺失补空记录并标记不一致。"""
    opencv = [record("a", is_pass=True), record("b", is_pass=False)]
    yolo = [record("b", is_pass=False), record("c", is_pass=True)]

    samples = align_samples(opencv, yolo)

    assert [s["name"] for s in samples] == ["a", "b", "c"]
    assert samples[1]["consistent"] is True
    assert samples[0]["consistent"] is False  # 仅 OpenCV 侧存在
    assert samples[2]["consistent"] is False
    assert samples[0]["yolo"]["total_count"] == 0


def test_build_summary():
    """汇总口径：平均耗时取两侧各自均值，一致率按样本计。"""
    samples = [
        {"consistent": True, "opencv": record("a", time_ms=10), "yolo": record("a", time_ms=20)},
        {"consistent": False, "opencv": record("b", is_pass=False, time_ms=30), "yolo": record("b", is_pass=False, time_ms=40)},
    ]
    summary = build_summary(samples)

    assert summary["opencv_avg_ms"] == 20
    assert summary["yolo_avg_ms"] == 30
    assert summary["opencv_pass"] == 1  # 两张样本中仅 a 为 PASS
    assert summary["yolo_pass"] == 1
    assert summary["opencv_total_count"] == 2
    assert summary["yolo_total_count"] == 2
    assert summary["consistency"] == "1/2（50.0%）"


def test_build_summary_excludes_warmup():
    """首帧预热不计入平均 detect 耗时。"""
    warm = record("a", time_ms=100.0)
    warm["warmup"] = True
    warm_yolo = record("a", time_ms=200.0)
    warm_yolo["warmup"] = True
    samples = [
        {"consistent": True, "opencv": warm, "yolo": warm_yolo},
        {"consistent": True, "opencv": record("b", time_ms=10.0), "yolo": record("b", time_ms=20.0)},
    ]

    summary = build_summary(samples)

    assert summary["opencv_avg_ms"] == 10.0
    assert summary["yolo_avg_ms"] == 20.0


def test_export_comparison_html(tmp_path):
    """渲染不报错，且左右两栏、一致标签、耗时、内嵌图齐备。"""
    output = export_comparison_html(sample_data(tmp_path), tmp_path / "algorithm_comparison.html")
    text = output.read_text(encoding="utf-8")

    assert output.exists()
    assert "crazing_98" in text and "crazing_148" in text
    assert "不一致" in text and "一致" in text
    assert "OpenCV" in text and "YOLO" in text
    assert "5.50 ms" in text and "17.50 ms" in text
    assert "data:image/jpeg;base64," in text
    assert 'src="http://' not in text and 'src="https://' not in text
    assert "判定一致率" in text and "1/2（50.0%）" in text
    assert "<th>PASS</th>" in text and "<th>FAIL</th>" in text
    assert "平均 detect 耗时" in text


def test_export_comparison_html_missing_image(tmp_path):
    """图片缺失时渲染占位，不抛异常。"""
    data = sample_data(tmp_path)
    data["samples"][0]["opencv"]["image_path"] = str(tmp_path / "not_exists.jpg")

    text = export_comparison_html(data, tmp_path / "compare.html").read_text(encoding="utf-8")

    assert "图片缺失" in text
