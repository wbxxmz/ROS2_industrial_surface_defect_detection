"""程序入口：python main.py --config demo.yaml"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inputs.factory import build_source  # noqa: E402
from pipeline.pipeline import Pipeline, build_detector  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.exceptions import ProjectError  # noqa: E402
from utils.logger import setup_logging  # noqa: E402


def resolve_config(path: str) -> Path:
    """定位配置文件：先按给定路径，再回退到项目根目录 / configs/。"""
    config_path = Path(path)
    if config_path.exists():
        return config_path
    for candidate in (PROJECT_ROOT / config_path, PROJECT_ROOT / "configs" / config_path):
        if candidate.exists():
            return candidate
    return config_path


def main(argv=None) -> int:
    """加载配置并跑完整条流水线；异常统一捕获为可读报错。"""
    parser = argparse.ArgumentParser(description="工业产品表面缺陷检测平台 · 阶段 1 骨架")
    parser.add_argument("--config", default="demo.yaml", help="配置文件路径（YAML/JSON）")
    args = parser.parse_args(argv)

    logger = setup_logging()
    try:
        config_path = resolve_config(args.config)
        config = load_config(config_path)
        logger.info("已加载配置：%s", config_path)

        detector = build_detector(config)
        source = build_source(config.get("input") or {})
        pipeline = Pipeline(config, detector, source)
        results = pipeline.run()
        logger.info("运行完成，共 %d 条检测结果", len(results))
        return 0
    except ProjectError as exc:  # 运行错误：契约规定的自定义异常
        print(f"[错误] 运行错误：{exc}", file=sys.stderr)
        return 1
    except ValueError as exc:  # 配置错误：配置非法（如未知 detector.type）
        print(f"[错误] 配置错误：{exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:  # 配置文件不存在
        logger.error("配置文件不存在：%s", exc)
        print(f"[错误] 配置文件不存在：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
