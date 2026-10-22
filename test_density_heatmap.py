"""临时验证脚本（扩展 02 第一步）：YOLO 检测 + 空间密度热力图。

用法：
    python3 test_density_heatmap.py [图片路径]

默认读取 ``test_picture/crazing_98.jpg``，输出 ``reports/visual/density_<stem>.jpg``。
同时验证「空检测列表直接返回、不画图、不报错」的边界行为。
"""

import logging
import sys
from pathlib import Path

import cv2
import yaml

from core import Frame
from pipeline.pipeline import build_detector
from visualization.heatmap import generate_density_heatmap

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "yolo_demo.yaml"
DEFAULT_IMAGE = PROJECT_ROOT / "test_picture" / "crazing_98.jpg"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "visual"


def main() -> int:
    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMAGE
    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[ERROR] 图片读取失败：{image_path}")
        return 1

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    detector = build_detector(config)
    detector.load(config.get("detector") or {})
    try:
        frame = Frame(
            image=image,
            width=image.shape[1],
            height=image.shape[0],
            frame_index=0,
            timestamp=0.0,
        )
        detections = detector.detect(frame)
        print(f"图片：{image_path.name}  尺寸：{image.shape[1]}×{image.shape[0]}")
        print(f"检出：{len(detections)} 条")
        for detection in detections:
            print(
                f"  - {detection.class_name:6s} conf={detection.confidence:.3f} "
                f"bbox={detection.bbox} area_ratio={detection.area_ratio:.4f}"
                f" severity={detection.severity}"
            )

        output = OUTPUT_DIR / f"density_{image_path.stem}.jpg"
        saved = generate_density_heatmap(frame, detections, output)
        print(f"密度热力图：{saved}")

        empty = generate_density_heatmap(frame, [], OUTPUT_DIR / "density_empty.jpg")
        print(f"空检测边界：返回值={empty!r}，未生成文件={not (OUTPUT_DIR / 'density_empty.jpg').exists()}")
    finally:
        detector.release()
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())
