"""阶段 4 WBS 1~2：NEU-DET 数据集整理（VOC→YOLO）+ YOLOv8 微调训练。

做的事：
1. 读 ``shapes/NEU-DET/IMAGES`` + ``ANNOTATIONS``（Pascal VOC XML）；
2. 按 ``NEU_TO_PROJECT`` 把 NEU 的 6 类合并为项目契约的 4 类（阶段 3 §4.1 映射表）；
3. 按类别分层切分 train/val（8:2），转成 YOLO txt 标注，落到 ``datasets/neu_det``；
4. 自动生成 ``neu_det.yaml``；
5. 用 ``yolov8m.pt`` 预训练权重微调 100 epochs（默认 device=0，强制走 CUDA GPU）；
6. 训练结束把 ``best.pt`` 复制到项目根目录。

用法：python3 train_yolo.py
"""

import argparse
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detectors.opencv_detector import CLASS_MAP  # noqa: E402  阶段 3 §4.1 唯一权威映射表

DATASET_ROOT = PROJECT_ROOT / "shapes" / "NEU-DET"
IMAGES_DIR = DATASET_ROOT / "IMAGES"
ANNOTATIONS_DIR = DATASET_ROOT / "ANNOTATIONS"
WORK_DIR = PROJECT_ROOT / "datasets" / "neu_det"  # 已在 .gitignore 屏蔽
YAML_PATH = PROJECT_ROOT / "neu_det.yaml"  # 含本机绝对路径，已在 .gitignore 屏蔽
PRETRAINED_NAME = "yolov8m.pt"  # 本地不存在时由 ultralytics 自动下载
PRETRAINED = PROJECT_ROOT / PRETRAINED_NAME
OUTPUT_WEIGHT = PROJECT_ROOT / "best.pt"
RUNS_DIR = PROJECT_ROOT / "runs"

# NEU-DET 原生 6 类 → 项目契约 4 类（语义就近合并，保留全部 1800 张）
# 依据：scratch=划痕、dent=凹坑、stain=污渍、notch=缺口（阶段 3 §4.1）
NEU_TO_PROJECT = {
    "scratches": 0,        # 划痕 → scratch
    "crazing": 0,          # 龟裂（网状细裂纹）→ scratch
    "pitted_surface": 1,   # 点蚀表面 → dent
    "patches": 2,          # 斑块 → stain
    "inclusion": 2,        # 夹杂物 → stain
    "rolled-in_scale": 3,  # 氧化铁皮压入 → notch
}
NAMES = [CLASS_MAP[i] for i in sorted(CLASS_MAP)]  # 顺序即 class_id，禁止重排

VAL_RATIO = 0.2
SEED = 42


def pick_device(requested: str) -> str:
    """校验并返回设备：``auto`` 按 CUDA 可用性选择，显式 GPU 不可用时给出清晰错误。"""
    import torch

    if requested == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    if requested != "cpu" and not requested.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            f"[错误] 指定 device={requested}，但当前 PyTorch 未启用 CUDA"
            f"（torch={torch.__version__}, cuda_available=False）。\n"
            "请先安装 NVIDIA 专有驱动 + CUDA 版 PyTorch，或改用 --device cpu。"
        )
    return requested


def parse_voc(xml_path: Path):
    """解析单个 VOC 标注，返回 (width, height, [(class_id, x1, y1, x2, y2), ...])。"""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width, height = int(size.findtext("width")), int(size.findtext("height"))
    objects = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        class_id = NEU_TO_PROJECT.get(name)
        if class_id is None:  # 未登记的类别名：直接报错，避免静默丢标注
            raise ValueError(f"{xml_path.name} 含未知类别 {name!r}，请先在 NEU_TO_PROJECT 中登记")
        box = obj.find("bndbox")
        objects.append(
            (
                class_id,
                int(float(box.findtext("xmin"))),
                int(float(box.findtext("ymin"))),
                int(float(box.findtext("xmax"))),
                int(float(box.findtext("ymax"))),
            )
        )
    return width, height, objects


def to_yolo_lines(width: int, height: int, objects) -> list:
    """VOC 的 x1y1x2y2 像素 → YOLO 的 cx cy w h 归一化；clamp 到图像内，丢弃退化框。"""
    lines = []
    for class_id, x1, y1, x2, y2 in objects:
        x1, y1 = max(0, min(width, x1)), max(0, min(height, y1))
        x2, y2 = max(0, min(width, x2)), max(0, min(height, y2))
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue
        cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w / width:.6f} {h / height:.6f}")
    return lines


def stratified_split(image_paths):
    """按类别前缀分层切分，保证每类都有验证样本；同 SEED 结果可复现。"""
    groups = {}
    for path in image_paths:
        groups.setdefault(path.stem.rsplit("_", 1)[0], []).append(path)

    rng = random.Random(SEED)
    train, val = [], []
    for key in sorted(groups):
        paths = sorted(groups[key])
        rng.shuffle(paths)
        n_val = max(1, round(len(paths) * VAL_RATIO))
        val.extend(paths[:n_val])
        train.extend(paths[n_val:])
    return sorted(train), sorted(val)


def build_dataset(force: bool = False) -> dict:
    """VOC→YOLO 转换 + 分层切分；已生成且数量一致时跳过（便于中断后重跑）。"""
    if not IMAGES_DIR.is_dir() or not ANNOTATIONS_DIR.is_dir():
        raise SystemExit(f"[错误] 数据集目录不完整：需要 {IMAGES_DIR} 与 {ANNOTATIONS_DIR}")

    images = sorted(p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if not images:
        raise SystemExit(f"[错误] 没有图片：{IMAGES_DIR}")

    train_paths, val_paths = stratified_split(images)
    expected_labels = len(train_paths) + len(val_paths)
    existing = list((WORK_DIR / "labels").rglob("*.txt")) if (WORK_DIR / "labels").is_dir() else []
    if existing and len(existing) == expected_labels and not force:
        print(f"[跳过转换] 已存在 {len(existing)} 份标注，如需重建请加 --rebuild-labels")
        return {"train": len(train_paths), "val": len(val_paths), "skipped": True}

    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)  # 全量重建，避免残留旧标注

    stats = {"train": 0, "val": 0, "boxes": 0, "skipped_no_box": 0, "missing_xml": 0}
    for split, paths in (("train", train_paths), ("val", val_paths)):
        (WORK_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (WORK_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
        for image_path in paths:
            xml_path = ANNOTATIONS_DIR / f"{image_path.stem}.xml"
            if not xml_path.is_file():
                stats["missing_xml"] += 1
                continue
            width, height, objects = parse_voc(xml_path)
            lines = to_yolo_lines(width, height, objects)
            if not lines:  # 映射后无有效框：不入库，避免训练时全为负样本
                stats["skipped_no_box"] += 1
                continue
            shutil.copy2(image_path, WORK_DIR / "images" / split / image_path.name)
            (WORK_DIR / "labels" / split / f"{image_path.stem}.txt").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )
            stats[split] += 1
            stats["boxes"] += len(lines)

    print(
        f"[转换完成] train={stats['train']} val={stats['val']} boxes={stats['boxes']} "
        f"无有效框跳过={stats['skipped_no_box']} 缺XML={stats['missing_xml']} → {WORK_DIR}"
    )
    return stats


def write_yaml() -> Path:
    """生成 ultralytics 数据集描述文件（nc/names 与阶段 3 §4.1 映射表逐项一致）。"""
    lines = [
        "# 自动生成，请勿手改；由 train_yolo.py 依据阶段 3 §4.1 映射表写出",
        f"path: {WORK_DIR}",
        "train: images/train",
        "val: images/val",
        "",
        f"nc: {len(NAMES)}",
        "names:",
    ]
    lines += [f"  {i}: {name}" for i, name in enumerate(NAMES)]
    YAML_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[数据集配置] {YAML_PATH}")
    return YAML_PATH


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="NEU-DET → YOLOv8 微调训练")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数（默认 100）")
    parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸（原图 200×200，默认 640）")
    parser.add_argument("--batch", type=int, default=0, help="批大小，0=按设备自动")
    parser.add_argument("--workers", type=int, default=-1, help="DataLoader 进程数，-1=按设备自动")
    parser.add_argument("--device", default="0", help="默认 0（GPU）；也可 auto / cpu / cuda:0")
    parser.add_argument("--rebuild-labels", action="store_true", help="强制重建 YOLO 标注")
    args = parser.parse_args(argv)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[错误] 未安装 ultralytics：python3 -m pip install ultralytics", file=sys.stderr)
        return 1

    device = pick_device(args.device)
    is_cpu = device == "cpu"
    batch = args.batch or (8 if is_cpu else 8)  # yolov8m@640 在 8GB 显存下 batch=8 稳妥，避免 OOM
    workers = args.workers if args.workers >= 0 else (2 if is_cpu else 4)

    build_dataset(force=args.rebuild_labels)
    data_yaml = write_yaml()

    weights = str(PRETRAINED) if PRETRAINED.is_file() else PRETRAINED_NAME
    print(f"[训练启动] weights={weights} device={device} epochs={args.epochs} "
          f"imgsz={args.imgsz} batch={batch} workers={workers}")

    model = YOLO(weights)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        workers=workers,
        device=device,
        project=str(RUNS_DIR),
        name="neu_det",
        exist_ok=True,
        seed=SEED,
        pretrained=True,
        verbose=True,
    )

    save_dir = Path(getattr(results, "save_dir", RUNS_DIR / "neu_det"))
    best = save_dir / "weights" / "best.pt"
    if not best.is_file():
        print(f"[错误] 训练结束但未找到 best.pt：{best}", file=sys.stderr)
        return 1
    shutil.copy2(best, OUTPUT_WEIGHT)
    print(f"[权重导出] {best} → {OUTPUT_WEIGHT}")

    metrics = getattr(results, "results_dict", None)
    if metrics:
        for key in ("metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/precision(B)", "metrics/recall(B)"):
            if key in metrics:
                print(f"[指标] {key} = {metrics[key]:.4f}")
    print(f"[完成] 训练产物目录：{save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
