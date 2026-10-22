"""阶段 7 §4.2：统计图生成（matplotlib，无 GUI）。

只产出 PNG 落盘，随后 ``plt.close(fig)`` 释放内存（§4.5）。
图内文字使用英文类别名（``CLASS_MAP`` 原值），避免中文字体依赖。
"""

import logging
import warnings
from pathlib import Path

# 系统存在多份 matplotlib 时，plt.subplots 会触发 Axes3D 导入告警；本项目只用 2D 柱状图，
# 定向过滤该条告警，避免污染用户终端（不影响其它告警）。
warnings.filterwarnings("ignore", message="Unable to import Axes3D.*", category=UserWarning)

logger = logging.getLogger("defect.report")


def class_distribution_chart(summary: dict, chart_path: Path, class_names: list) -> Path:
    """按类别绘制缺陷数量柱状图，返回 PNG 路径。"""
    import matplotlib

    matplotlib.use("Agg")  # 无头后端：不弹出窗口（§1 无头架构）
    import matplotlib.pyplot as plt

    counts = summary.get("class_counts") or {}
    labels = list(class_names)
    values = [int(counts.get(class_id, 0)) for class_id in sorted(counts)]

    fig, ax = plt.subplots(figsize=(6.0, 2.8), dpi=150)
    try:
        bars = ax.bar(labels, values, color=["#D64541", "#3B7DD8", "#3FA34D", "#E58E26"][: len(labels)])
        ax.set_title("Defect Count by Class", fontsize=11)
        ax.set_ylabel("Count", fontsize=9)
        ax.tick_params(labelsize=8)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                str(value),
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_ylim(0, max(values + [1]) * 1.2)
        fig.tight_layout()
        fig.savefig(chart_path)
    finally:
        plt.close(fig)  # §4.5：无论成功失败都释放 Figure
    return chart_path
