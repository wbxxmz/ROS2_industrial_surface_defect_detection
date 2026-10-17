"""判定模块（实现文档 阶段5 · 4.2）。

默认规则：任一缺陷 confidence ≥ 阈值 且 area_px ≥ 最小面积 即判为不合格。
规则参数可配置，支持按类别设不同阈值。
"""

from typing import Optional

from defect_core.types import InspectionResult


class Judge:
    """单件产品合格 / 不合格判定器。"""

    def __init__(self, cfg: Optional[dict] = None) -> None:
        cfg = cfg or {}
        self.confidence_threshold = float(cfg.get('confidence_threshold', 0.5))
        self.min_area_px = int(cfg.get('min_area_px', 50))
        # 可选的按类别覆盖阈值，如 {'scratch': {'min_area_px': 100}}
        self.per_class = cfg.get('per_class', {})

    def __call__(self, result: InspectionResult) -> bool:
        """返回 True 表示合格；任一缺陷触发规则即返回 False。"""
        for det in result.defects:
            override = self.per_class.get(det.class_name, {})
            threshold = override.get('confidence_threshold', self.confidence_threshold)
            min_area = override.get('min_area_px', self.min_area_px)
            if det.confidence >= threshold and det.area_px >= min_area:
                return False
        return True
