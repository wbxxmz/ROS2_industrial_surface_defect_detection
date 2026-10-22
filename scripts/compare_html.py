"""V1.1 扩展：双算法对比 HTML —— 同一批图片分别跑 OpenCV / YOLO，输出可视化对比页。

用法：
    python3 scripts/compare_html.py
    python3 scripts/compare_html.py --input test_picture --output reports/algorithm_comparison.html

流程：
1. 用 ``--opencv-config`` / ``--yolo-config`` 各跑一遍 ``Pipeline``（仅覆盖 input.dir）；
2. 逐样本记录标注图、判定、检出数、max_severity、单帧 ``detect()`` 耗时；
3. 用 YOLO 模型为每张样本生成 EigenCAM 注意力热力图（仅展示，不参与判定）；
4. 按文件名主干（stem）对齐两组结果，交给 ``comparison_html`` 渲染自包含 HTML。
"""

import argparse
import copy
import shutil
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402

from core import Frame  # noqa: E402
from inputs.factory import build_source  # noqa: E402
from pipeline.pipeline import Pipeline, build_detector  # noqa: E402
from report.exporters.comparison_html import export_comparison_html  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.exceptions import DetectorInferenceError  # noqa: E402
from visualization.attention import generate_eigencam_heatmap  # noqa: E402

# 两种算法的标注图分开存放，避免同名文件互相覆盖
COMPARE_VISUAL_DIR = PROJECT_ROOT / "reports" / "visual" / "compare"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


class TimedPipeline(Pipeline):
    """仅扩展耗时采集：记录每帧 ``detect()`` 的毫秒耗时，其余逻辑复用 Pipeline。"""

    def __init__(self, config, detector, source) -> None:
        super().__init__(config, detector, source)
        self.detect_ms: list[float] = []

    def detect(self, frame):
        start = time.perf_counter()
        detections = super().detect(frame)
        self.detect_ms.append((time.perf_counter() - start) * 1000.0)
        return detections


def run_algorithm(label: str, config: dict, input_dir: str) -> list[dict]:
    """按给定 config 跑一遍流水线，返回逐样本记录（已把标注图复制到独立目录）。"""
    config = copy.deepcopy(config)
    config.setdefault("input", {})["dir"] = input_dir

    detector = build_detector(config)
    source = build_source(config.get("input") or {})
    pipeline = TimedPipeline(config, detector, source)
    pipeline.run()

    COMPARE_VISUAL_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for sample, elapsed_ms in zip(pipeline.samples, pipeline.detect_ms):
        name = sample["source"]
        target = COMPARE_VISUAL_DIR / f"{name}__{label}.jpg"
        image_path = None
        try:
            shutil.copy2(sample["image_path"], target)
            image_path = str(target)
        except OSError as exc:  # 复制失败降级为占位，不影响对比页
            print(f"[警告] 标注图复制失败：{sample['image_path']}（{exc}）")
        verdict = sample["verdict"]
        records.append(
            {
                "name": name,
                "image_path": image_path,
                "is_pass": verdict.is_pass,
                "total_count": verdict.total_count,  # 计入判定的有效缺陷数
                "max_severity": verdict.max_severity,
                "time_ms": elapsed_ms,
                "warmup": not records,  # 首帧含预热开销，不计入平均耗时（沿用压测脚本口径）
            }
        )
    return records


def align_samples(opencv_records: list[dict], yolo_records: list[dict]) -> list[dict]:
    """按文件名主干对齐两算法结果；单侧缺失时补空记录并标记不一致。"""
    empty = {
        "image_path": None,
        "is_pass": False,
        "total_count": 0,
        "max_severity": None,
        "time_ms": 0.0,
        "warmup": False,
    }
    ocv_map = {record["name"]: record for record in opencv_records}
    yolo_map = {record["name"]: record for record in yolo_records}

    samples = []
    for name in sorted(set(ocv_map) | set(yolo_map)):
        left = ocv_map.get(name, empty)
        right = yolo_map.get(name, empty)
        samples.append(
            {
                "name": name,
                "consistent": left["is_pass"] == right["is_pass"] and name in ocv_map and name in yolo_map,
                "opencv": left,
                "yolo": right,
            }
        )
    return samples


def generate_attention_maps(config: dict, input_dir: str, records: list[dict]) -> None:
    """为 YOLO 记录补 ``attention_path``：EigenCAM 热力图写入对比目录。

    单张失败（读图失败 / 推理异常）只打印警告并留空，页面降级为占位块。
    """
    names = {record["name"] for record in records}
    if not names:
        return

    COMPARE_VISUAL_DIR.mkdir(parents=True, exist_ok=True)
    root = Path(input_dir)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    if not root.is_dir():  # 视频/摄像头输入无逐张原图，跳过注意力图
        print(f"[警告] 输入不是图片目录，跳过注意力热力图：{root}")
        return

    detector = build_detector(config)
    detector.load(config.get("detector") or {})
    try:
        for path in sorted(root.iterdir()):
            if path.stem not in names or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            image = cv2.imread(str(path))
            if image is None:
                print(f"[警告] 注意力热力图跳过（读图失败）：{path}")
                continue
            frame = Frame(
                image=image,
                width=image.shape[1],
                height=image.shape[0],
                frame_index=0,
                timestamp=0.0,
            )
            try:
                output = generate_eigencam_heatmap(
                    frame, detector, COMPARE_VISUAL_DIR / f"{path.stem}__attention.jpg"
                )
            except (DetectorInferenceError, OSError) as exc:  # 单张失败不影响整页
                print(f"[警告] 注意力热力图生成失败：{path.stem}（{exc}）")
                continue
            for record in records:
                if record["name"] == path.stem:
                    record["attention_path"] = output
    finally:
        detector.release()


def _steady_times(samples: list[dict], side: str) -> list[float]:
    """取剔除首帧预热后的单帧耗时（无有效样本时回退为全量）。"""
    steady = [sample[side]["time_ms"] for sample in samples if not sample[side].get("warmup")]
    return steady or [sample[side]["time_ms"] for sample in samples]


def build_summary(samples: list[dict]) -> dict:
    """汇总平均 detect 耗时（剔除首帧预热）、总检出数、PASS/FAIL 数与判定一致率。"""
    ocv_times = _steady_times(samples, "opencv")
    yolo_times = _steady_times(samples, "yolo")
    same = sum(1 for sample in samples if sample["consistent"])
    total = len(samples)
    return {
        "opencv_avg_ms": statistics.fmean(ocv_times) if ocv_times else 0.0,
        "yolo_avg_ms": statistics.fmean(yolo_times) if yolo_times else 0.0,
        "opencv_total_count": sum(sample["opencv"]["total_count"] for sample in samples),
        "yolo_total_count": sum(sample["yolo"]["total_count"] for sample in samples),
        "opencv_pass": sum(1 for sample in samples if sample["opencv"]["is_pass"]),
        "yolo_pass": sum(1 for sample in samples if sample["yolo"]["is_pass"]),
        "consistency": f"{same}/{total}（{same / total * 100:.1f}%）" if total else "0/0（0.0%）",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="OpenCV vs YOLO 双算法对比 HTML")
    parser.add_argument("--opencv-config", default="configs/demo.yaml", help="OpenCV 检测器配置")
    parser.add_argument("--yolo-config", default="configs/yolo_demo.yaml", help="YOLO 检测器配置")
    parser.add_argument("--input", default="test_picture", help="样本目录（覆盖两个 config 的 input.dir）")
    parser.add_argument("--output", default="reports/algorithm_comparison.html", help="HTML 输出路径")
    args = parser.parse_args(argv)

    opencv_config = load_config(PROJECT_ROOT / args.opencv_config)
    yolo_config = load_config(PROJECT_ROOT / args.yolo_config)

    print(f"样本目录：{args.input}")
    print("OpenCV 推理中 ...")
    opencv_records = run_algorithm("opencv", opencv_config, args.input)
    print("YOLO 推理中 ...")
    yolo_records = run_algorithm("yolo", yolo_config, args.input)
    print("YOLO 注意力热力图生成中 ...")
    generate_attention_maps(yolo_config, args.input, yolo_records)

    samples = align_samples(opencv_records, yolo_records)
    summary = build_summary(samples)
    data = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_dir": args.input,
        "samples": samples,
        "summary": summary,
    }

    output = export_comparison_html(data, PROJECT_ROOT / args.output)
    print("")
    print("=== 汇总 ===")
    print(f"样本数：{len(samples)}")
    print(
        f"OpenCV：平均 {summary['opencv_avg_ms']:.2f} ms/帧，总检出 {summary['opencv_total_count']}，"
        f"PASS {summary['opencv_pass']} / FAIL {len(samples) - summary['opencv_pass']}"
    )
    print(
        f"YOLO  ：平均 {summary['yolo_avg_ms']:.2f} ms/帧，总检出 {summary['yolo_total_count']}，"
        f"PASS {summary['yolo_pass']} / FAIL {len(samples) - summary['yolo_pass']}"
    )
    print(f"判定一致率：{summary['consistency']}")
    print(f"对比页：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
