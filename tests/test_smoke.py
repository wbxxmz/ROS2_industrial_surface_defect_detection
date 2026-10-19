"""冒烟测试：验证 main.py 能跑通整条流水线（阶段 1 §3.8 / §7）。"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import main as main_module  # noqa: E402


def test_smoke_pipeline_runs():
    """python main.py --config demo.yaml 能跑通、不崩溃（返回 0）。"""
    assert main_module.main(["--config", "demo.yaml"]) == 0
