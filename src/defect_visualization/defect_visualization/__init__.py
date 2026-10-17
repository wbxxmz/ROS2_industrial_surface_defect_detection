"""defect_visualization：叠绘、判定与统计。"""

from .judge import Judge
from .overlay import draw_detections, draw_stats

__all__ = ['draw_detections', 'draw_stats', 'Judge']
