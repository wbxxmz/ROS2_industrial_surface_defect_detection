"""阶段 7 §5：ReportBuilder —— 质检报告生成唯一对外入口。

职责边界：
- 数据组装：校验 ``batch``、读取 ``Verdict`` / ``Detection``、调用阶段 5 的
  ``summarize()`` 求批次汇总（报告不重复计算，保证与统计面板逐字段一致）；
- 渲染导出：委托私有导出器写 PDF / Excel，模板与数据分离（§4.2）。

异常语义（§4.4）：
- ``ValueError``：``batch`` 为 None / 非 list / 空列表 / 元素非 dict / 缺必填键 /
  ``verdict`` 类型不符；或 ``cfg`` 必填项缺失或 ``format`` 不受支持。
- ``OSError``：目标目录无法创建或无写权限、磁盘空间不足等（底层异常原样上抛，不吞）。
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List

from core.types import Detection
from core.verdict import Verdict
from detectors.opencv_detector import CLASS_MAP  # 阶段 3 §4.1 唯一权威映射表
from judge import summarize

logger = logging.getLogger("defect.report")

REQUIRED_KEYS = ("source", "verdict", "detections")
SUPPORTED_FORMATS = ("pdf", "xlsx")
SEVERITY_LABELS = {0: "轻度", 1: "中度", 2: "重度"}


def severity_label(value: object) -> str:
    """严重度的中文展示：None → "未分级"（不得当作 0），0/1/2 → 轻度/中度/重度。"""
    if value is None:
        return "未分级"
    return SEVERITY_LABELS.get(int(value), f"S{value}")


def _validate_batch(batch: object) -> List[dict]:
    """按 §4.4 校验 batch；任一不合法即抛 ``ValueError``。"""
    if batch is None or not isinstance(batch, list):
        raise ValueError(f"batch 必须是 list，收到 {type(batch).__name__}")
    if not batch:
        raise ValueError("batch 为空列表：不生成空报告（阶段 7 §4.4）")
    for index, item in enumerate(batch):
        if not isinstance(item, dict):
            raise ValueError(f"batch[{index}] 必须是 dict，收到 {type(item).__name__}")
        missing = [key for key in REQUIRED_KEYS if key not in item]
        if missing:
            raise ValueError(f"batch[{index}] 缺少必填键 {missing}（阶段 7 §5）")
        if not isinstance(item["verdict"], Verdict):
            raise ValueError(f"batch[{index}]['verdict'] 不是 Verdict：{type(item['verdict']).__name__}")
        if not isinstance(item["detections"], list):
            raise ValueError(f"batch[{index}]['detections'] 必须是 list")
        for det in item["detections"]:
            if not isinstance(det, Detection):
                raise ValueError(f"batch[{index}]['detections'] 含非 Detection 元素")
    return batch


def _validate_cfg(cfg: object) -> dict:
    """校验 cfg 必填项；``format`` 统一转小写。"""
    if not isinstance(cfg, dict):
        raise ValueError(f"cfg 必须是 dict，收到 {type(cfg).__name__}")
    output_path = cfg.get("output_path")
    if not isinstance(output_path, str) or not output_path:
        raise ValueError("cfg.output_path 必须是非空 str")
    fmt = str(cfg.get("format") or "").lower()
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"cfg.format 只支持 {SUPPORTED_FORMATS}，收到 {cfg.get('format')!r}")
    return {**cfg, "format": fmt}


class ReportBuilder:
    """质检报告构建器：``build(batch, cfg) -> Path``。"""

    def build(self, batch: List[dict], cfg: dict) -> Path:
        """组装并导出报告，返回落盘路径（阶段 7 §5）。"""
        samples = _validate_batch(batch)
        config = _validate_cfg(cfg)
        data = self._assemble(samples, config)

        output = Path(str(config["output_path"])).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)  # 无法创建时 OSError 原样上抛

        if config["format"] == "pdf":
            from report.exporters.pdf_exporter import export_pdf

            export_pdf(data, output)
        else:
            from report.exporters.xlsx_exporter import export_xlsx

            export_xlsx(data, output)

        logger.info("报告已生成：%s（格式 %s，样本 %d）", output, config["format"], len(samples))
        return output

    def _assemble(self, samples: List[dict], cfg: dict) -> dict:
        """把 batch 组装成导出器可直接消费的纯数据（不持有 Verdict 之外的中间对象）。"""
        verdicts = [item["verdict"] for item in samples]
        summary = summarize(verdicts)  # 阶段 5 §4.3：唯一汇总口径

        severities = [v.max_severity for v in verdicts if v.max_severity is not None]
        overall_severity = max(severities) if severities else None

        # 与 judge() 的有效缺陷预筛保持同一口径（§4.2.1）：用于明细表「是否计入判定」
        judge_cfg = cfg.get("judge") or {}
        min_confidence = float(judge_cfg.get("min_confidence", 0.45))
        min_area_px = int(judge_cfg.get("min_area_px", 10))

        sample_rows = []
        detail_rows = []
        for item in samples:
            verdict = item["verdict"]
            image_path = item.get("image_path")
            sample_rows.append(
                {
                    "source": str(item["source"]),
                    "is_pass": verdict.is_pass,
                    "result": "合格" if verdict.is_pass else "不合格",
                    "reason": verdict.reason,
                    "total_count": verdict.total_count,
                    "max_severity": severity_label(verdict.max_severity),
                    "image_path": str(image_path) if image_path else None,
                }
            )
            for det in item["detections"]:
                x, y, w, h = det.bbox
                counted = (
                    det.class_id in CLASS_MAP
                    and det.confidence >= min_confidence
                    and det.area_px >= min_area_px
                )
                detail_rows.append(
                    {
                        "sample": str(item["source"]),
                        "class_name": det.class_name,
                        "class_id": det.class_id,
                        "bbox": f"({x}, {y}, {w}, {h})",
                        "area_px": int(det.area_px),
                        "area_ratio": float(det.area_ratio),
                        "confidence": float(det.confidence),
                        "severity": severity_label(det.severity),
                        "counted": "是" if counted else "否",
                        "counted_raw": counted,
                    }
                )

        return {
            "title": str(cfg.get("title") or "工业产品表面缺陷质检报告"),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "algorithm_name": str(cfg.get("algorithm_name") or "未知算法"),
            "algorithm_version": str(cfg.get("algorithm_version") or "unknown"),
            "params": cfg.get("params"),
            "font_path": cfg.get("font_path"),
            "sample_source": str(cfg.get("sample_source") or ""),
            "summary": summary,
            "overall_max_severity": severity_label(overall_severity),
            "class_names": [CLASS_MAP[class_id] for class_id in sorted(CLASS_MAP)],
            "samples": sample_rows,
            "details": detail_rows,
        }
