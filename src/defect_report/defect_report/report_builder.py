"""质检报告生成（实现文档 阶段7）。

当前为骨架：报告字段必须直接取自检测结果，避免人工手填造成口径漂移。
数据组装与渲染导出分离，便于后续新增报告样式。
"""

import json
from pathlib import Path
from typing import List

from defect_core.types import InspectionResult


class ReportBuilder:
    """把检测批次结果组装为结构化报告并导出。"""

    def __init__(self, cfg: dict = None) -> None:
        cfg = cfg or {}
        self.output_dir = Path(cfg.get('output_dir', 'reports'))
        self.formats = cfg.get('formats', ['json'])

    def build(self, results: List[InspectionResult], meta: dict = None) -> dict:
        """组装报告数据：概览 + 缺陷明细 + 统计 + 溯源信息。"""
        meta = meta or {}
        total = len(results)
        passed = sum(1 for r in results if r.is_pass)
        return {
            'meta': {
                **meta,
                # 溯源信息：算法名称/版本、时间等，保证报告可追溯
                'detector_name': results[0].detector_name if results else '',
                'detector_version': results[0].detector_version if results else '',
            },
            'summary': {
                'total_samples': total,
                'passed': passed,
                'failed': total - passed,
                'yield_rate': (passed / total) if total else 0.0,
            },
            'details': [self._sample_detail(r) for r in results],
        }

    @staticmethod
    def _sample_detail(result: InspectionResult) -> dict:
        return {
            'source_id': result.source_id,
            'is_pass': result.is_pass,
            'total_defects': result.total_defects,
            'defects': [
                {
                    'class_name': d.class_name,
                    'bbox': list(d.bbox),
                    'area_px': d.area_px,
                    'area_ratio': d.area_ratio,
                    'confidence': d.confidence,
                    'severity': d.severity,
                }
                for d in result.defects
            ],
        }

    def export(self, report: dict, name: str = 'quality_report') -> List[Path]:
        """导出报告文件，返回生成的文件路径列表。"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        outputs: List[Path] = []
        if 'json' in self.formats:
            path = self.output_dir / f'{name}.json'
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                            encoding='utf-8')
            outputs.append(path)
        # TODO(阶段7): PDF(reportlab) / Word(python-docx) / Excel(openpyxl) 导出与图表
        return outputs
