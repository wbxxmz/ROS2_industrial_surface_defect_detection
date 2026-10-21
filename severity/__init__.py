"""severity：阶段 6 §4.1 缺陷严重程度分级。"""

from severity.grader import (
    SEVERITY_MILD,
    SEVERITY_MODERATE,
    SEVERITY_NAMES,
    SEVERITY_SEVERE,
    SeverityConfig,
    grade,
)

__all__ = [
    "SEVERITY_MILD",
    "SEVERITY_MODERATE",
    "SEVERITY_SEVERE",
    "SEVERITY_NAMES",
    "SeverityConfig",
    "grade",
]
