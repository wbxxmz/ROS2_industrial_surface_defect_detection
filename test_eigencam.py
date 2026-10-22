"""临时验证脚本（扩展 02 第二步）：YOLO 推理 + EigenCAM 注意力热力图。

用法：
    python3 test_eigencam.py [图片路径] [层索引]
    python3 test_eigencam.py --all            # 批量跑 test_picture/ 全部图片

默认读取 ``test_picture/crazing_98.jpg``、使用第 6 层（backbone stage-4 C2f），
输出 ``reports/visual/attention_<stem>.jpg``。
"""

import logging
import sys
from pathlib import Path

import cv2
import yaml

from core import Frame
from pipeline.pipeline import build_detector
from visualization.attention import DEFAULT_LAYER_INDEX, generate_eigencam_heatmap

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "yolo_demo.yaml"
DEFAULT_IMAGE = PROJECT_ROOT / "test_picture" / "crazing_98.jpg"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "visual"


def _process(detector, image_path: Path, layer_name) -> bool:
    """对单张图生成注意力热力图；读图失败返回 False。"""
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[ERROR] 图片读取失败：{image_path}")
        return False

    frame = Frame(
        image=image,
        width=image.shape[1],
        height=image.shape[0],
        frame_index=0,
        timestamp=0.0,
    )
    output = OUTPUT_DIR / f"attention_{image_path.stem}.jpg"
    saved = generate_eigencam_heatmap(frame, detector, output, layer_name)
    print(f"{image_path.name}: {saved}")
    return True


def main() -> int:
    batch_mode = "--all" in sys.argv[1:]
    layer_name = DEFAULT_LAYER_INDEX
    if not batch_mode and len(sys.argv) > 2:
        layer_name = int(sys.argv[2])

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    detector = build_detector(config)
    detector.load(config.get("detector") or {})
    try:
        if batch_mode:
            for path in sorted((PROJECT_ROOT / "test_picture").glob("*.jpg")):
                _process(detector, path, None)
            forward_model = detector.inference_model or detector.model
            target = forward_model.model[DEFAULT_LAYER_INDEX]
            print(f"批量完成；残留 hook 数：{len(target._forward_hooks)}")
            return 0

        image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMAGE
        if not image_path.is_absolute():
            image_path = PROJECT_ROOT / image_path
        if not _process(detector, image_path, layer_name):
            return 1

        frame_image = cv2.imread(str(image_path))
        frame = Frame(
            image=frame_image,
            width=frame_image.shape[1],
            height=frame_image.shape[0],
            frame_index=0,
            timestamp=0.0,
        )
        # hook 摘除校验：生成后层上不应残留回调
        forward_model = detector.inference_model or detector.model
        target = forward_model.model[layer_name]
        print(f"残留 hook 数：{len(target._forward_hooks)}")
        # 污染校验：再次 detect() 应正常返回
        print(f"生成后 detect() 复跑：{len(detector.detect(frame))} 条（hook 未污染推理）")
    finally:
        detector.release()
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())
