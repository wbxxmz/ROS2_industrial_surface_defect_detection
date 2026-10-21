"""阶段 5 §5.1：判定结果数据结构 Verdict。

字段名冻结（阶段 7 直接消费），含义与可空性见阶段 5 §5.1 表。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Verdict:
    """单件产品的结构化判定结果。"""

    is_pass: bool  # 合格=True
    reason: str  # 人类可读，命中规则以 "；" 分隔；保证非空
    total_count: int  # 有效缺陷数，>= 0
    class_counts: Dict[int, int]  # 键恒为 0/1/2/3，缺省类计 0
    total_area_px: int  # 有效缺陷 area_px 之和
    total_area_ratio: float  # 有效缺陷 area_ratio 之和，clip 到 [0,1]
    triggered: List[str] = field(default_factory=list)  # 命中的规则 id，按 §4.2.1 顺序
    # 阶段 6 §4.1.5：有效缺陷中的最高严重度（0/1/2）；无有效缺陷或全部未分级时为 None
    max_severity: Optional[int] = None
