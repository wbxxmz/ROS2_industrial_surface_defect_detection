"""V1.1 扩展 01：HTML 交互式报告导出（自包含单文件，无外链资源）。

- 输入为 ``ReportBuilder._assemble()`` 产出的纯数据 dict，不重算统计口径；
- 统计图 / 标注图经 base64 内嵌，CSS 内联，断网也能打开；
- 所有文本字段渲染前统一 ``html.escape()``，防止 ``< > &`` 破坏页面结构；
- ``image_path`` 缺失时渲染灰底占位块 +「图片缺失」，不抛异常（与 PDF 导出器口径一致）。
"""

import base64
import html
import logging
from pathlib import Path

logger = logging.getLogger("defect.report")

MISSING_TEXT = "图片缺失"

_STYLE = """
<style>
:root { --bg:#F5F7FA; --card:#FFFFFF; --line:#DCE3EC; --ink:#1F2937; --muted:#6B7280; --pass:#1F9D55; --fail:#D64541; }
* { box-sizing: border-box; }
body { margin:0; padding:32px; background:var(--bg); color:var(--ink);
       font-family:-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; }
.wrap { max-width:1180px; margin:0 auto; }
h1 { font-size:24px; margin:0 0 6px; }
.sub { color:var(--muted); font-size:13px; margin-bottom:24px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:20px 22px; margin-bottom:20px; }
h2 { font-size:16px; margin:0 0 14px; }
table { width:100%; border-collapse:collapse; font-size:13px; }
th,td { border-bottom:1px solid var(--line); padding:8px 10px; text-align:left; vertical-align:middle; }
th { background:#F2F4F7; font-weight:600; }
tr.dim td { color:#9CA3AF; }
.kpis { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; }
.kpi { background:#F8FAFC; border:1px solid var(--line); border-radius:8px; padding:14px; }
.kpi .label { color:var(--muted); font-size:12px; }
.kpi .value { font-size:20px; font-weight:700; margin-top:6px; }
.pass { color:var(--pass); font-weight:600; }
.fail { color:var(--fail); font-weight:600; }
.gallery { display:grid; grid-template-columns:repeat(auto-fill,minmax(260px,1fr)); gap:16px; }
.shot { width:100%; border-radius:8px; display:block; background:#EEE; }
.missing { display:flex; align-items:center; justify-content:center; height:180px; color:var(--muted);
           font-size:13px; border:1px dashed #C9D1DB; background:#EEEEEE; }
.note { color:var(--muted); font-size:12px; margin-top:10px; }
.meta { color:var(--muted); font-size:12px; line-height:1.8; }
</style>
"""


def _escape(value) -> str:
    """文本字段统一转义：``None`` 视为空串，其余一律 ``str()`` 后转义。"""
    return html.escape("" if value is None else str(value), quote=True)


def _data_uri(path) -> "str | None":
    """读取本地图片并编码为 data URI；不可读 / 不存在返回 ``None``（不抛异常）。"""
    if not path:
        return None
    try:
        file_path = Path(str(path))
        if not file_path.is_file():
            return None
        mime = "image/png" if file_path.suffix.lower() == ".png" else "image/jpeg"
        return f"data:{mime};base64,{base64.b64encode(file_path.read_bytes()).decode('ascii')}"
    except OSError as exc:  # 读盘失败：降级为占位，不中断导出
        logger.warning("标注图读取失败：%s（%s）", path, exc)
        return None


def _image_block(path, alt: str) -> str:
    """图片可用则内嵌 <img>，否则渲染灰底占位块 +「图片缺失」。"""
    uri = _data_uri(path)
    if uri:
        return f'<img class="shot" src="{uri}" alt="{_escape(alt)}">'
    logger.warning("标注图缺失或不存在：%s（渲染占位）", path)
    return f'<div class="shot missing">{MISSING_TEXT}</div>'


def export_html(data: dict, output: Path) -> Path:
    """把组装好的报告数据写成自包含 HTML，返回落盘路径。"""
    from report.charts import class_distribution_chart

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    summary = data["summary"]

    # 统计图：复用阶段 7 的无头柱状图，转 base64 后删除临时文件
    chart_path = output.parent / f"{output.stem}_class_chart.png"
    class_distribution_chart(summary, chart_path, data["class_names"])
    try:
        chart_block = _image_block(chart_path, "缺陷类别分布")
    finally:
        chart_path.unlink(missing_ok=True)

    kpis = [
        ("样本总数", summary["sample_count"]),
        ("合格 / 不合格", f"{summary['pass_count']} / {summary['fail_count']}"),
        ("良率", f"{summary['yield_rate'] * 100:.2f}%"),
        ("缺陷总数", summary["total_count"]),
        ("缺陷总面积占比", f"{summary['total_area_ratio'] * 100:.2f}%"),
        ("批次最高严重度", data["overall_max_severity"]),
    ]
    kpi_html = "".join(
        f'<div class="kpi"><div class="label">{_escape(label)}</div>'
        f'<div class="value">{_escape(value)}</div></div>'
        for label, value in kpis
    )

    overview_rows = [
        ("检测批次时间", data["generated_at"]),
        ("算法", f"{data['algorithm_name']} {data['algorithm_version']}"),
        ("样本来源", data["sample_source"] or "未提供"),
        ("参数", data["params"] if data["params"] is not None else "未提供"),
    ]
    overview_html = "".join(
        f"<tr><th>{_escape(key)}</th><td>{_escape(value)}</td></tr>" for key, value in overview_rows
    )

    if data["details"]:
        detail_rows = []
        for row in data["details"]:
            row_class = "" if row["counted_raw"] else ' class="dim"'
            detail_rows.append(
                f"<tr{row_class}>"
                f"<td>{_escape(row['sample'])}</td>"
                f"<td>{_escape(row['class_name'])}</td>"
                f"<td>{_escape(row['bbox'])}</td>"
                f"<td>{float(row['area_ratio']) * 100:.2f}%</td>"
                f"<td>{float(row['confidence']):.2f}</td>"
                f"<td>{_escape(row['severity'])}</td>"
                f"<td>{_escape(row['counted'])}</td>"
                "</tr>"
            )
        detail_html = "".join(detail_rows)
    else:
        detail_html = '<tr><td colspan="7">本批次未检出有效缺陷（明细表为空）。</td></tr>'

    cards = []
    for sample in data["samples"]:
        media = _image_block(sample.get("image_path"), f"{sample['source']} 标注图")
        state = "pass" if sample["is_pass"] else "fail"
        cards.append(
            '<div class="card">'
            f"{media}"
            f'<div style="margin-top:10px;font-size:13px;">{_escape(sample["source"])}</div>'
            f'<div class="meta">判定：<span class="{state}">{_escape(sample["result"])}</span>　'
            f"缺陷数：{_escape(sample['total_count'])}　最高严重度：{_escape(sample['max_severity'])}</div>"
            f'<div class="meta">原因：{_escape(sample["reason"])}</div>'
            "</div>"
        )

    page = (
        "<!DOCTYPE html>\n"
        '<html lang="zh-CN">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{_escape(data['title'])}</title>\n"
        f"{_STYLE}\n</head>\n<body>\n<div class=\"wrap\">\n"
        f"<h1>{_escape(data['title'])}</h1>\n"
        f"<div class=\"sub\">生成时间：{_escape(data['generated_at'])} ｜ "
        f"算法：{_escape(data['algorithm_name'])} {_escape(data['algorithm_version'])}</div>\n"
        f'<div class="card"><div class="kpis">{kpi_html}</div></div>\n'
        '<div class="card"><h2>一、检测概览</h2>'
        f'<table>{overview_html}</table>'
        '<div class="note">* 明细包含所有原始检出，其中未达置信度阈值的框不计入最终判定（灰色行）。</div>'
        "</div>\n"
        '<div class="card"><h2>二、缺陷类别分布</h2>'
        f"{chart_block}</div>\n"
        '<div class="card"><h2>三、缺陷明细</h2><table>'
        "<thead><tr><th>样本</th><th>类别</th><th>位置 (x, y, w, h)</th>"
        "<th>面积占比</th><th>置信度</th><th>严重度</th><th>计入判定</th></tr></thead>"
        f"<tbody>{detail_html}</tbody></table></div>\n"
        '<div class="card"><h2>四、样本抽检图</h2>'
        f'<div class="gallery">{"".join(cards)}</div></div>\n'
        "</div>\n</body>\n</html>\n"
    )

    output.write_text(page, encoding="utf-8")
    logger.info("HTML 报告已生成：%s（样本 %d）", output, len(data["samples"]))
    return output
