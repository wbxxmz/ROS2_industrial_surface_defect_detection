"""图像级检出率评估（可复现产物）：对同一批图片逐张跑 detect + judge，落 CSV + Markdown。

口径与 ``scripts/compare_algorithms.py`` 完全一致：
- 阈值全部来自 ``configs/*.yaml``（不改 ``min_confidence`` / ``min_area_px`` / ``max_area_ratio`` / ``max_count``）；
- 判定复用 ``judge.judge``，``有效检出数`` 即 ``Verdict.total_count``（已按 class_id / min_confidence / min_area_px 预筛）；
- 计时口径一致：先用第 1 张做一次预热推理（不计时），再对每张计时 ``detect()``。

与 ``compare_algorithms.py`` 的唯一差别：**逐图异常隔离**。某张图抛异常时记为 ``ERROR``、
不计入耗时/框数统计、并在报告里单列，避免一张坏图中断整批（例如 OpenCV 在自交轮廓上报
``convexityDefects`` 错误）。

用法：
    python3 scripts/eval_image_level.py                       # 默认 val 360 张，两个算法
    python3 scripts/eval_image_level.py --limit 30            # 只取前 30 张（文件名排序）
    python3 scripts/eval_image_level.py --algo yolo           # 只跑 YOLO
"""

import argparse
import csv
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
DEFAULT_DATA_DIR = PROJECT_ROOT / "datasets" / "neu_det" / "images" / "val"
OUTPUT_DIR = PROJECT_ROOT / "reports"

# 算法 → (显示名, 检测器配置, 判定配置来源)；与 compare_algorithms.py 的默认值保持一致
ALGO_SPECS = {
    "opencv": ("OpenCV", "configs/demo.yaml"),
    "yolo": ("YOLO(best.pt)", "configs/yolo_demo.yaml"),
}
JUDGE_CONFIG = "configs/demo.yaml"
CSV_NAME = "image_level_val360.csv"
MD_NAME = "image_level_val360.md"

#: 逐图备注（键：(算法显示名, 文件名)）。只用于在产物里标注"需要人工跟进"的读数，
#: 不参与统计与判定，也不会改变任何数值；写死在这里是为了重跑后备注不丢。
ROW_NOTES: dict[tuple[str, str], str] = {
    (
        "OpenCV",
        "rolled-in_scale_146.jpg",
    ): "修复后新增 4 条 dent（bbox 宽/高均 <10px，面积 14–28 px²），未人工核验",
}


def pick_images(data_dir: Path, limit: int | None) -> list[Path]:
    """按文件名排序取图片；``limit`` 为空表示全量。"""
    if not data_dir.is_dir():
        raise SystemExit(f"[错误] 数据目录不存在：{data_dir}")
    paths = sorted(p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTS)
    if not paths:
        raise SystemExit(f"[错误] 数据目录中没有图片：{data_dir}")
    return paths if limit is None else paths[:limit]


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


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def run_algo(algo: str, frames: list[tuple[Path, Frame]], judge_cfg: dict) -> tuple[list[dict], dict]:
    """跑单个算法，返回逐图行与汇总。"""
    label, config_rel = ALGO_SPECS[algo]
    config = load_config(PROJECT_ROOT / config_rel)
    detector = build_detector(config)
    detector.load(config.get("detector") or {})

    rows: list[dict] = []
    errors: list[tuple[str, str]] = []
    try:
        if frames:  # 预热：与 compare_algorithms.py 一致，第 1 张不计时
            detector.detect(frames[0][1])
        for path, frame in frames:
            try:
                start = time.perf_counter()
                detections = detector.detect(frame)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
            except Exception as exc:  # 逐图隔离：坏图不中断整批
                errors.append((path.name, f"{type(exc).__name__}: {exc}"))
                rows.append(
                    {
                        "name": path.name,
                        "algo": label,
                        "count": 0,
                        "valid_count": 0,
                        "is_fail": "ERROR",
                        "max_conf": "",
                        "ms": "",
                        "reason": f"{type(exc).__name__}: {exc}",
                        "confs": [],
                    }
                )
                continue

            verdict = judge(detections, judge_cfg)
            rows.append(
                {
                    "name": path.name,
                    "algo": label,
                    "count": len(detections),
                    "valid_count": verdict.total_count,
                    "is_fail": "FAIL" if not verdict.is_pass else "PASS",
                    "max_conf": round(max((d.confidence for d in detections), default=0.0), 3)
                    if detections
                    else "",
                    "ms": round(elapsed_ms, 2),
                    "reason": verdict.reason,
                    "confs": [round(d.confidence, 3) for d in detections],
                }
            )
    finally:
        detector.release()

    evaluated = [row for row in rows if row["is_fail"] != "ERROR"]
    times = [row["ms"] for row in evaluated]
    counts = [row["count"] for row in evaluated]
    fail = sum(1 for row in evaluated if row["is_fail"] == "FAIL")
    summary = {
        "label": label,
        "config": config_rel,
        "sample_total": len(rows),
        "evaluated": len(evaluated),
        "errors": errors,
        "fail": fail,
        "pass": len(evaluated) - fail,
        "fail_rate": (fail / len(evaluated) * 100.0) if evaluated else 0.0,
        "zero_detect": sum(1 for row in evaluated if row["count"] == 0),
        "boxes": sum(counts),
        "valid_boxes": sum(row["valid_count"] for row in evaluated),
        "mean_boxes": mean(counts),
        "mean_ms": mean(times),
        "median_ms": statistics.median(times) if times else 0.0,
        "max_ms": max(times) if times else 0.0,
        "pass_rows": [row for row in evaluated if row["is_fail"] == "PASS"],
    }
    return rows, summary


def write_csv(rows: list[dict]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / CSV_NAME
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["文件名", "算法", "检出数", "有效检出数", "是否FAIL", "最高conf", "耗时ms", "备注"])
        for row in rows:
            writer.writerow(
                [
                    row["name"],
                    row["algo"],
                    row["count"],
                    row["valid_count"],
                    row["is_fail"],
                    row["max_conf"],
                    row["ms"],
                    ROW_NOTES.get((row["algo"], row["name"]), ""),
                ]
            )
    return path


def write_markdown(summaries: list[dict], data_dir: Path, judge_cfg: dict, limit: int | None) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / MD_NAME
    lines = [
        "# 图像级检出率评估（val 全量）",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 数据目录：`{data_dir}`",
        f"- 样本数：{summaries[0]['sample_total']}"
        + (f"（`--limit {limit}`）" if limit is not None else "（全量）"),
        f"- 判定配置：`{JUDGE_CONFIG}` → `max_count={judge_cfg.get('max_count')}`, "
        f"`max_area_ratio={judge_cfg.get('max_area_ratio')}`, "
        f"`min_confidence={judge_cfg.get('min_confidence')}`, "
        f"`min_area_px={judge_cfg.get('min_area_px')}`",
        "- 口径：与 `scripts/compare_algorithms.py` 一致——阈值全部取自 YAML；判定复用 `judge.judge`，"
        "`有效检出数` = `Verdict.total_count`（已按 `class_id` / `min_confidence` / `min_area_px` 预筛）；"
        "计时先用第 1 张预热（不计时），随后逐张计 `detect()` 耗时。",
        "- 唯一差别：逐图异常隔离——抛异常的行记为 `ERROR`，不计入耗时/框数统计，并在下文单列。",
        "",
        "## 汇总",
        "",
        "| 指标 | " + " | ".join(summary["label"] for summary in summaries) + " |",
        "| --- | " + " | ".join("---" for _ in summaries) + " |",
        "| 检测器配置 | " + " | ".join(f"`{s['config']}`" for s in summaries) + " |",
        "| 样本数 | " + " | ".join(str(s["sample_total"]) for s in summaries) + " |",
        "| 已评估（排除 ERROR） | " + " | ".join(str(s["evaluated"]) for s in summaries) + " |",
        "| FAIL | " + " | ".join(str(s["fail"]) for s in summaries) + " |",
        "| PASS | " + " | ".join(str(s["pass"]) for s in summaries) + " |",
        "| 图像级检出率（FAIL/已评估） | "
        + " | ".join(f"{s['fail_rate']:.1f}%" for s in summaries)
        + " |",
        "| 零检出图 | " + " | ".join(str(s["zero_detect"]) for s in summaries) + " |",
        "| 总框数（原始检出） | " + " | ".join(str(s["boxes"]) for s in summaries) + " |",
        "| 有效框数（judge 预筛后） | " + " | ".join(str(s["valid_boxes"]) for s in summaries) + " |",
        "| 平均框/帧 | " + " | ".join(f"{s['mean_boxes']:.2f}" for s in summaries) + " |",
        "| detect 均值 | " + " | ".join(f"{s['mean_ms']:.2f} ms" for s in summaries) + " |",
        "| detect 中位 | " + " | ".join(f"{s['median_ms']:.2f} ms" for s in summaries) + " |",
        "| detect 最大 | " + " | ".join(f"{s['max_ms']:.2f} ms" for s in summaries) + " |",
        "| 异常图 | " + " | ".join(str(len(s["errors"])) for s in summaries) + " |",
        "",
        "## 判 PASS 的图（逐条）",
        "",
        "| 算法 | 图片 | 检出数 | 有效检出数 | 最高 conf | 各框 conf | 判定原因 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    pass_rows = [(summary["label"], row) for summary in summaries for row in summary["pass_rows"]]
    if pass_rows:
        for label, row in pass_rows:
            confs = ", ".join(f"{c:.3f}" for c in row["confs"]) or "—"
            lines.append(
                f"| {label} | {row['name']} | {row['count']} | {row['valid_count']} | "
                f"{row['max_conf']} | {confs} | {row['reason']} |"
            )
    else:
        lines.append("| — | （无） | — | — | — | — | — |")

    lines.extend(["", "## 异常图（ERROR，已从统计中排除）", ""])
    error_rows = [(summary["label"], name, message) for summary in summaries for name, message in summary["errors"]]
    if error_rows:
        lines.append("| 算法 | 图片 | 异常 |")
        lines.append("| --- | --- | --- |")
        for label, name, message in error_rows:
            lines.append(f"| {label} | {name} | `{message}` |")
    else:
        lines.append("（无）")

    lines.extend(["", "## 逐图备注（人工跟进项，不参与统计与判定）", ""])
    labels = {summary["label"] for summary in summaries}
    note_rows = [
        (label, name, note) for (label, name), note in ROW_NOTES.items() if label in labels
    ]
    if note_rows:
        lines.append("| 算法 | 图片 | 备注 |")
        lines.append("| --- | --- | --- |")
        for label, name, note in note_rows:
            lines.append(f"| {label} | {name} | {note} |")
    else:
        lines.append("（无）")

    lines.extend(
        [
            "",
            "## 产物与复现",
            "",
            f"- 逐图明细：`{OUTPUT_DIR / CSV_NAME}`"
            "（列：文件名, 算法, 检出数, 有效检出数, 是否FAIL, 最高conf, 耗时ms, 备注）",
            "- 复现：`python3 scripts/eval_image_level.py --algo both`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="图像级检出率评估（只读图片，只写 reports/）")
    parser.add_argument("--limit", type=int, default=None, help="只评估前 N 张（按文件名排序），默认全量")
    parser.add_argument("--algo", choices=("yolo", "opencv", "both"), default="both", help="评估哪个算法")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="图片目录，默认 val 全量")
    args = parser.parse_args(argv)

    if args.limit is not None and args.limit <= 0:
        raise SystemExit("[错误] --limit 必须为正整数")

    data_dir = Path(args.data_dir)
    paths = pick_images(data_dir, args.limit)
    frames = [(path, load_frame(path)) for path in paths]
    judge_cfg = load_config(PROJECT_ROOT / JUDGE_CONFIG).get("judge") or {}

    algos = ["opencv", "yolo"] if args.algo == "both" else [args.algo]
    print(f"数据目录：{data_dir}")
    print(f"样本数：{len(frames)}")

    all_rows: list[dict] = []
    summaries: list[dict] = []
    for algo in algos:
        print(f"{ALGO_SPECS[algo][0]} 推理中 ...")
        rows, summary = run_algo(algo, frames, judge_cfg)
        all_rows.extend(rows)
        summaries.append(summary)

    csv_path = write_csv(all_rows)
    md_path = write_markdown(summaries, data_dir, judge_cfg, args.limit)

    print("")
    print("=== 汇总 ===")
    for summary in summaries:
        print(
            f"{summary['label']}：图像级 FAIL {summary['fail']}/{summary['evaluated']} "
            f"= {summary['fail_rate']:.1f}%，零检出 {summary['zero_detect']}，"
            f"总框 {summary['boxes']}（有效 {summary['valid_boxes']}，{summary['mean_boxes']:.2f}/帧），"
            f"detect 均值 {summary['mean_ms']:.2f} ms / 中位 {summary['median_ms']:.2f} ms，"
            f"异常 {len(summary['errors'])}"
        )
    print(f"CSV：{csv_path}")
    print(f"报告：{md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
