"""report：质检报告（阶段 7 实现）。

对外唯一入口 ``ReportBuilder``（阶段 7 §5）；导出器为私有实现。
"""

from report.builder import ReportBuilder, severity_label

__all__ = ["ReportBuilder", "severity_label"]
