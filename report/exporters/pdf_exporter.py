"""阶段 7 §4.2：PDF 导出（reportlab，无头落盘）。

- 中文渲染：优先用 ``cfg.font_path``，否则自动探测系统 TrueType 中文字体；
  探测不到时退回 Helvetica 并记 warning（§8 已知风险，不中断导出）。
- 图片缺失：渲染灰底占位「图片缺失 / N/A」，不抛异常、不中断报告（§4.4）。
"""

import io
import logging
from pathlib import Path

logger = logging.getLogger("defect.report")

# 系统可用的 TrueType 中文字体（reportlab 不支持 CFF 轮廓的 .ttc，故排除 NotoSansCJK）
# 注意：DroidSansFallbackFull 在 reportlab 子集化下 ASCII 会渲染成方框，故排在最后兜底
FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/usr/share/fonts/truetype/arphic/ukai.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
)
PLACEHOLDER_SIZE = 120  # 缺失图片占位方块的边长（pt）


def _register_font(font_path):
    """注册中文字体，返回可用字体名；失败退回 Helvetica 并记 warning。"""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    candidates = [font_path] if font_path else []
    candidates += [path for path in FONT_CANDIDATES if path != font_path]
    for candidate in candidates:
        if not candidate or not Path(candidate).is_file():
            continue
        try:
            kwargs = {"subfontIndex": 0} if str(candidate).lower().endswith(".ttc") else {}
            pdfmetrics.registerFont(TTFont("ReportCJK", str(candidate), **kwargs))
            return "ReportCJK"
        except Exception as exc:  # 字体格式不支持：换下一个候选
            logger.warning("中文字体不可用：%s（%s）", candidate, exc)
    logger.warning("未找到可用中文字体：PDF 中文可能乱码（阶段 7 §8）")
    return "Helvetica"


def _placeholder(text):
    """缺失图片占位：浅灰背景 + 文字提示。"""
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph, Table

    style = ParagraphStyle("placeholder", fontName="ReportCJK", fontSize=8, alignment=1, textColor=colors.grey)
    cell = Table(
        [[Paragraph(text, style)]],
        colWidths=[PLACEHOLDER_SIZE],
        rowHeights=[PLACEHOLDER_SIZE],
        style=[("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EEEEEE")),
               ("VALIGN", (0, 0), (-1, -1), "MIDDLE")],
    )
    return cell


def export_pdf(data: dict, output: Path) -> Path:
    """把组装好的数据写成 PDF，返回路径（§5）。"""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        Image,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    from report.charts import class_distribution_chart

    font_name = _register_font(data.get("font_path"))
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("title", parent=styles["Title"], fontName=font_name, fontSize=18, leading=24)
    heading_style = ParagraphStyle("heading", parent=styles["Heading2"], fontName=font_name, fontSize=12, leading=16)
    body_style = ParagraphStyle("body", parent=styles["Normal"], fontName=font_name, fontSize=9, leading=13)
    cell_style = ParagraphStyle("cell", parent=body_style, fontSize=7.5, leading=10)
    small_style = ParagraphStyle("small", parent=body_style, fontSize=7, leading=9, textColor=colors.grey)

    # 图片统一以内存字节流交给 reportlab：避免其内部长期持有文件句柄（§4.5 无句柄泄漏）
    buffers = []

    def image_from_bytes(raw: bytes, width, height):
        buffer = io.BytesIO(raw)
        buffers.append(buffer)
        return Image(buffer, width=width, height=height)

    chart_path = output.parent / f"{output.stem}_class_chart.png"
    class_distribution_chart(data["summary"], chart_path, data["class_names"])
    chart_bytes = chart_path.read_bytes()
    chart_path.unlink(missing_ok=True)  # 临时统计图不残留

    doc = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=1.6 * cm,
        rightMargin=1.6 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        title=data["title"],
    )

    summary = data["summary"]
    story = [Paragraph(data["title"], title_style), Spacer(1, 0.3 * cm)]

    overview = [
        ["检测批次时间", data["generated_at"], "样本总数", str(summary["sample_count"])],
        ["合格 / 不合格", f"{summary['pass_count']} / {summary['fail_count']}", "良率",
         f"{summary['yield_rate'] * 100:.2f}%"],
        ["缺陷总数", str(summary["total_count"]), "缺陷总面积占比",
         f"{summary['total_area_ratio'] * 100:.2f}%"],
        ["算法", f"{data['algorithm_name']} {data['algorithm_version']}", "批次最高严重度",
         data["overall_max_severity"]],
    ]
    overview_table = Table(overview, colWidths=[3.0 * cm, 4.6 * cm, 3.4 * cm, 4.6 * cm])
    overview_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), font_name),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BBBBBB")),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F2F4F7")),
                ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#F2F4F7")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story += [Paragraph("一、检测概览", heading_style), overview_table]
    story.append(
        Paragraph(
            "* 明细包含所有原始检出，其中未达置信度阈值的框不计入最终判定。",
            small_style,
        )
    )
    story.append(Spacer(1, 0.4 * cm))

    story += [Paragraph("二、缺陷类别分布", heading_style)]
    story.append(image_from_bytes(chart_bytes, 12.5 * cm, 5.8 * cm))
    story.append(Spacer(1, 0.4 * cm))

    story += [Paragraph("三、缺陷明细", heading_style)]
    header = ["样本", "类别", "位置 (x, y, w, h)", "面积占比", "置信度", "严重度", "计入判定"]
    rows = [header] + [
        [
            row["sample"],
            row["class_name"],
            row["bbox"],
            f"{row['area_ratio'] * 100:.2f}%",
            f"{row['confidence']:.2f}",
            row["severity"],
            row["counted"],
        ]
        for row in data["details"]
    ]
    detail_table = Table(
        rows,
        colWidths=[3.9 * cm, 1.9 * cm, 3.6 * cm, 1.9 * cm, 1.6 * cm, 1.7 * cm, 1.7 * cm],
        repeatRows=1,
    )
    detail_style = [
        ("FONTNAME", (0, 0), (-1, -1), font_name),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BBBBBB")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE3EC")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]
    for offset, row in enumerate(data["details"], start=1):
        if not row["counted_raw"]:  # 未计入判定：整行浅灰，与有效缺陷在视觉上区分
            detail_style.append(("TEXTCOLOR", (0, offset), (-1, offset), colors.grey))
    detail_table.setStyle(TableStyle(detail_style))
    story.append(detail_table)
    story.append(Spacer(1, 0.3 * cm))
    if not data["details"]:
        story.append(Paragraph("本批次未检出有效缺陷（明细表为空）。", body_style))

    story += [PageBreak(), Paragraph("四、样本抽检图", heading_style)]
    image_cells = []
    for sample in data["samples"]:
        path = sample["image_path"]
        if path and Path(path).is_file():
            visual = image_from_bytes(Path(path).read_bytes(), 7.6 * cm, 7.6 * cm)
        else:
            logger.warning("样本 %s 的标注图缺失或不存在：%s（渲染占位）", sample["source"], path)
            visual = _placeholder("图片缺失 / N/A")
        label = Paragraph(
            f"{sample['source']}<br/>判定：{sample['result']}　缺陷数：{sample['total_count']}　"
            f"最高严重度：{sample['max_severity']}",
            cell_style,
        )
        image_cells.append([visual, label])

    for start in range(0, len(image_cells), 2):  # 两列排布，控制页数
        pair = image_cells[start : start + 2]
        if len(pair) == 1:
            pair.append([_placeholder("无样本"), Paragraph("", cell_style)])
        grid = Table(pair, colWidths=[8.0 * cm, 8.0 * cm])
        grid.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story += [grid, Spacer(1, 0.2 * cm)]

    story += [Spacer(1, 0.4 * cm), Paragraph("五、溯源信息", heading_style)]
    story.append(
        Paragraph(
            f"算法：{data['algorithm_name']}（版本 {data['algorithm_version']}）<br/>"
            f"生成时间：{data['generated_at']}<br/>"
            f"样本来源：{data['sample_source'] or '未提供'}<br/>"
            f"参数：{data['params'] if data['params'] is not None else '未提供'}",
            small_style,
        )
    )

    try:
        doc.build(story)
    finally:
        for buffer in buffers:  # §4.5：导出结束释放内存缓冲
            buffer.close()
    return output
