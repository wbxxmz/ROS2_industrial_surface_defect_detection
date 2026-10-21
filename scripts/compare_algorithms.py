"""阶段 5 后基线对比：OpenCV 传统算法 vs YOLO(best.pt) 批量跑同一批图片。

只取数据、不画图：
- 逐图记录检测数量、单帧耗时；
- 两算法共用同一份 judge 配置，记录 is_pass / reason；
- 输出 Markdown 报告（默认 ``comparison_report.md``）。

用法：
    python3 scripts/compare_algorithms.py
    python3 scripts/compare_algorithms.py --limit 30 --output comparison_report.md
"""

import argparse
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import Frame  # noqa: E402
from judge import judge  # noqa: E402
from pipeline.pipeline import build_detector  # noqa: E402
from utils.config import load_config  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
# 验证集优先；兼容两种常见布局，最后回退到训练集
DATA_CANDIDATES = (
    PROJECT_ROOT / "datasets" / "neu_det" / "images" / "val",
    PROJECT_ROOT / "datasets" / "neu_det" / "val" / "images",
    PROJECT_ROOT / "datasets" / "neu_det" / "images" / "train",
)


def pick_images(limit: int, strategy: str = "stratified") -> tuple[Path, list[Path]]:
    """定位数据集目录并取样本。

    ``stratified`` 按文件名前缀（NEU-DET 的类别名）轮转取样，避免只落到同一类；
    ``head`` 直接取排序后的前 ``limit`` 张。
    """
    for data_dir in DATA_CANDIDATES:
        if not data_dir.is_dir():
            continue
        paths = sorted(p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTS)
        if not paths:
            continue
        if strategy == "head":
            return data_dir, paths[:limit]

        groups: dict[str, list[Path]] = {}
        for path in paths:
            groups.setdefault(path.stem.split("_")[0], []).append(path)
        if len(groups) == 1:
            return data_dir, paths[:limit]

        selected: list[Path] = []
        cursors = {key: 0 for key in groups}
        keys = sorted(groups)
        while len(selected) < limit and any(cursors[k] < len(groups[k]) for k in keys):
            for key in keys:
                if len(selected) >= limit:
                    break
                if cursors[key] < len(groups[key]):
                    selected.append(groups[key][cursors[key]])
                    cursors[key] += 1
        return data_dir, selected
    raise SystemExit(f"[错误] 未找到 NEU-DET 图片目录，已尝试：{[str(p) for p in DATA_CANDIDATES]}")


def load_frame(path: Path) -> Frame:
    image = cv2.imread(str(path))
    if image is None:
        raise SystemExit(f"[错误] 图片不可读：{path}")
    return Frame(
        image=image,
        width=int(image.shape[1]),
        height=int(image.shape[0]),
        frame_index=0,
        timestamp=0.0,
    )


def run_detector(detector, frames: list[Frame]) -> tuple[list[list], list[float]]:
    """先做一次预热推理（不计时），再逐帧计时；返回每帧检测结果与毫秒耗时。"""
    if frames:
        detector.detect(frames[0])
    results: list[list] = []
    times_ms: list[float] = []
    for frame in frames:
        start = time.perf_counter()
        detections = detector.detect(frame)
        times_ms.append((time.perf_counter() - start) * 1000.0)
        results.append(detections)
    return results, times_ms


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def build_markdown(
    rows: list[dict], data_dir: Path, limit: int, judge_cfg: dict, sample_strategy: str
) -> str:
    opencv_pass = sum(1 for row in rows if row["opencv_is_pass"])
    yolo_pass = sum(1 for row in rows if row["yolo_is_pass"])
    total = len(rows)
    same = sum(1 for row in rows if row["opencv_is_pass"] == row["yolo_is_pass"])
    both_pass = sum(1 for row in rows if row["opencv_is_pass"] and row["yolo_is_pass"])
    both_fail = sum(1 for row in rows if not row["opencv_is_pass"] and not row["yolo_is_pass"])
    ocv_pass_yolo_fail = sum(1 for row in rows if row["opencv_is_pass"] and not row["yolo_is_pass"])
    ocv_fail_yolo_pass = sum(1 for row in rows if not row["opencv_is_pass"] and row["yolo_is_pass"])

    ocv_times = [row["opencv_ms"] for row in rows]
    yolo_times = [row["yolo_ms"] for row in rows]
    ocv_counts = [row["opencv_count"] for row in rows]
    yolo_counts = [row["yolo_count"] for row in rows]

    lines = [
        "# 双算法对比报告（OpenCV vs YOLO best.pt）",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 数据集：`{data_dir}`（取样方式：{sample_strategy}，共 {limit} 张）",
        f"- 判定配置：`max_count={judge_cfg.get('max_count')}`, "
        f"`max_area_ratio={judge_cfg.get('max_area_ratio')}`, "
        f"`min_confidence={judge_cfg.get('min_confidence')}`, "
        f"`min_area_px={judge_cfg.get('min_area_px')}`（两算法共用）",
        "- 说明：仅统计检测数据，不绘制标注图；耗时为单帧 `detect()` 调用，已排除模型加载与首次预热。",
        "",
        "## 汇总",
        "",
        "| 指标 | OpenCV | YOLO(best.pt) |",
        "| --- | --- | --- |",
        f"| 样本数 | {total} | {total} |",
        f"| PASS | {opencv_pass} | {yolo_pass} |",
        f"| FAIL | {total - opencv_pass} | {total - yolo_pass} |",
        f"| 平均检测数/帧 | {mean(ocv_counts):.2f} | {mean(yolo_counts):.2f} |",
        f"| 平均单帧耗时 | {mean(ocv_times):.1f} ms | {mean(yolo_times):.1f} ms |",
        f"| 中位单帧耗时 | {statistics.median(ocv_times):.1f} ms | {statistics.median(yolo_times):.1f} ms |",
        f"| 最大单帧耗时 | {max(ocv_times):.1f} ms | {max(yolo_times):.1f} ms |",
        "",
        "## 判定一致性",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        f"| 一致样本数 | {same}/{total} |",
        f"| 一致率 | {same / total * 100:.1f}% |",
        f"| 同为 PASS | {both_pass} |",
        f"| 同为 FAIL | {both_fail} |",
        f"| OpenCV PASS → YOLO FAIL | {ocv_pass_yolo_fail} |",
        f"| OpenCV FAIL → YOLO PASS | {ocv_fail_yolo_pass} |",
        "",
        "## 逐图明细",
        "",
        "| # | 图片 | OpenCV 检测数 | OpenCV 耗时(ms) | OpenCV 判定 | YOLO 检测数 | YOLO 耗时(ms) | YOLO 判定 | 一致 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for index, row in enumerate(rows, start=1):
        lines.append(
            f"| {index} | {row['name']} | {row['opencv_count']} | {row['opencv_ms']:.1f} | "
            f"{row['opencv_label']} | {row['yolo_count']} | {row['yolo_ms']:.1f} | "
            f"{row['yolo_label']} | {'是' if row['opencv_is_pass'] == row['yolo_is_pass'] else '否'} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="OpenCV vs YOLO 批量对比（只统计，不画图）")
    parser.add_argument("--limit", type=int, default=30, help="样本数量，默认 30")
    parser.add_argument("--opencv-config", default="configs/demo.yaml", help="OpenCV 检测器配置")
    parser.add_argument("--yolo-config", default="configs/yolo_demo.yaml", help="YOLO 检测器配置")
    parser.add_argument("--judge-config", default="configs/demo.yaml", help="判定配置来源（取其中 judge 段）")
    parser.add_argument("--output", default="comparison_report.md", help="Markdown 报告输出路径")
    parser.add_argument(
        "--sample-strategy",
        choices=("stratified", "head"),
        default="stratified",
        help="取样方式：stratified 按类别轮转（默认），head 取排序前 N 张",
    )
    args = parser.parse_args(argv)

    if args.limit <= 0:
        raise SystemExit("[错误] --limit 必须为正整数")

    opencv_config = load_config(PROJECT_ROOT / args.opencv_config)
    yolo_config = load_config(PROJECT_ROOT / args.yolo_config)
    judge_cfg = load_config(PROJECT_ROOT / args.judge_config).get("judge") or {}

    data_dir, paths = pick_images(args.limit, args.sample_strategy)
    frames = [load_frame(path) for path in paths]
    print(f"数据集：{data_dir}")
    print(f"样本数：{len(frames)}")

    opencv_detector = build_detector(opencv_config)
    yolo_detector = build_detector(yolo_config)
    load_start = time.perf_counter()
    opencv_detector.load(opencv_config.get("detector") or {})
    yolo_detector.load(yolo_config.get("detector") or {})
    load_ms = (time.perf_counter() - load_start) * 1000.0

    try:
        print(f"模型加载耗时（两个检测器合计）：{load_ms:.0f} ms")
        print("OpenCV 推理中 ...")
        opencv_results, opencv_times = run_detector(opencv_detector, frames)
        print("YOLO 推理中 ...")
        yolo_results, yolo_times = run_detector(yolo_detector, frames)
    finally:
        opencv_detector.release()
        yolo_detector.release()

    rows: list[dict] = []
    for path, ocv_dets, ocv_ms, yolo_dets, yolo_ms in zip(
        paths, opencv_results, opencv_times, yolo_results, yolo_times
    ):
        ocv_verdict = judge(ocv_dets, judge_cfg)
        yolo_verdict = judge(yolo_dets, judge_cfg)
        rows.append(
            {
                "name": path.name,
                "opencv_count": len(ocv_dets),
                "opencv_ms": ocv_ms,
                "opencv_is_pass": ocv_verdict.is_pass,
                "opencv_label": "PASS" if ocv_verdict.is_pass else "FAIL",
                "yolo_count": len(yolo_dets),
                "yolo_ms": yolo_ms,
                "yolo_is_pass": yolo_verdict.is_pass,
                "yolo_label": "PASS" if yolo_verdict.is_pass else "FAIL",
            }
        )

    total = len(rows)
    opencv_pass = sum(1 for row in rows if row["opencv_is_pass"])
    yolo_pass = sum(1 for row in rows if row["yolo_is_pass"])
    same = sum(1 for row in rows if row["opencv_is_pass"] == row["yolo_is_pass"])
    ocv_times = [row["opencv_ms"] for row in rows]
    yolo_times = [row["yolo_ms"] for row in rows]

    print("")
    print("=== 最终统计 ===")
    print(f"样本数：{total}")
    print(f"OpenCV：PASS {opencv_pass} / FAIL {total - opencv_pass}，平均 {mean(ocv_times):.1f} ms/帧")
    print(f"YOLO  ：PASS {yolo_pass} / FAIL {total - yolo_pass}，平均 {mean(yolo_times):.1f} ms/帧")
    print(f"判定一致：{same}/{total} = {same / total * 100:.1f}%")
    print(f"  - 同为 PASS：{sum(1 for r in rows if r['opencv_is_pass'] and r['yolo_is_pass'])}")
    print(f"  - 同为 FAIL：{sum(1 for r in rows if not r['opencv_is_pass'] and not r['yolo_is_pass'])}")
    print(f"  - OpenCV PASS → YOLO FAIL：{sum(1 for r in rows if r['opencv_is_pass'] and not r['yolo_is_pass'])}")
    print(f"  - OpenCV FAIL → YOLO PASS：{sum(1 for r in rows if not r['opencv_is_pass'] and r['yolo_is_pass'])}")

    output_path = PROJECT_ROOT / args.output
    output_path.write_text(
        build_markdown(rows, data_dir, total, judge_cfg, args.sample_strategy), encoding="utf-8"
    )
    print(f"报告：{output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
