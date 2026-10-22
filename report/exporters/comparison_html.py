"""V1.1 扩展：双算法对比 HTML 导出（自包含单文件）。

复用 ``html_exporter`` 的样式常量与转义 / base64 工具，只负责对比页面的结构编排。
"""

import logging
from pathlib import Path

from report.exporters.html_exporter import _STYLE, _data_uri, _escape

logger = logging.getLogger("defect.report")

MISSING_TEXT = "图片缺失"

_EXTRA_STYLE = """
<style>
.grid2 { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
.side { border:1px solid var(--line); border-radius:8px; padding:12px; }
.side h3 { font-size:13px; margin:0 0 10px; color:var(--muted); font-weight:600; }
.badge { display:inline-block; padding:2px 10px; border-radius:999px; font-size:12px; font-weight:600; margin-left:8px; }
.badge.same { background:#E6F4EA; color:var(--pass); }
.badge.diff { background:#FDECEA; color:var(--fail); }
.shot.small { height:auto; }
.kv { display:flex; justify-content:space-between; font-size:13px; padding:4px 0; color:var(--muted); }
.kv b { color:var(--ink); }
</style>
"""


def _image_block(path, alt: str) -> str:
    """图片可用则 base64 内嵌，否则灰底占位（不抛异常）。"""
    uri = _data_uri(path)
    if uri:
        return f'<img class="shot" src="{uri}" alt="{_escape(alt)}">'
    return f'<div class="shot missing">{MISSING_TEXT}</div>'


def _fmt_severity(value) -> str:
    """严重度展示：None 显示为「—」，否则 S0/S1/S2。"""
    return "—" if value is None else f"S{value}"


def _side_block(title: str, record: dict) -> str:
    """单算法一栏：标注图 + 判定 + 检出数 + 最高严重度 + 耗时。"""
    state = "pass" if record["is_pass"] else "fail"
    result = "PASS" if record["is_pass"] else "FAIL"
    return (
        '<div class="side">'
        f"<h3>{_escape(title)}</h3>"
        f'{_image_block(record.get("image_path"), f"{title} 标注图")}'
        f'<div class="kv" style="margin-top:10px;">判定：<b class="{state}">{result}</b></div>'
        f'<div class="kv">检出数<b>{_escape(record.get("total_count", 0))}</b></div>'
        f'<div class="kv">最高严重度<b>{_escape(_fmt_severity(record.get("max_severity")))}</b></div>'
        f'<div class="kv">单帧耗时<b>{float(record.get("time_ms", 0.0)):.2f} ms'
        f'{"（预热）" if record.get("warmup") else ""}</b></div>'
        "</div>"
    )


def export_comparison_html(data: dict, output: Path) -> Path:
    """把对比数据渲染为自包含 HTML，返回落盘路径。"""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    summary = data["summary"]
    samples = data["samples"]

    summary_rows = []
    for algorithm, avg_ms, total_count, pass_count in (
        ("OpenCV 传统算法", summary["opencv_avg_ms"], summary["opencv_total_count"], summary["opencv_pass"]),
        ("YOLO", summary["yolo_avg_ms"], summary["yolo_total_count"], summary["yolo_pass"]),
    ):
        fail_count = len(samples) - pass_count
        summary_rows.append(
            "<tr>"
            f"<td>{_escape(algorithm)}</td>"
            f"<td>{float(avg_ms):.2f} ms</td>"
            f"<td>{_escape(total_count)}</td>"
            f'<td class="pass">{_escape(pass_count)}</td>'
            f'<td class="fail">{_escape(fail_count)}</td>'
            "</tr>"
        )
    summary_rows = "".join(summary_rows)

    cards = []
    for sample in samples:
        same = sample["consistent"]
        badge = (
            '<span class="badge same">一致</span>'
            if same
            else '<span class="badge diff">不一致</span>'
        )
        cards.append(
            '<div class="card">'
            f'<div style="font-size:14px;font-weight:600;margin-bottom:12px;">'
            f"{_escape(sample['name'])}{badge}</div>"
            f'<div class="grid2">{_side_block("OpenCV", sample["opencv"])}'
            f'{_side_block("YOLO", sample["yolo"])}</div>'
            "</div>"
        )

    page = (
        "<!DOCTYPE html>\n"
        '<html lang="zh-CN">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>双算法检测对比</title>\n"
        f"{_STYLE}\n{_EXTRA_STYLE}\n</head>\n<body>\n<div class=\"wrap\">\n"
        "<h1>双算法检测对比：OpenCV vs YOLO</h1>\n"
        f'<div class="sub">生成时间：{_escape(data["generated_at"])} ｜ '
        f'样本来源：{_escape(data["input_dir"])} ｜ 样本数：{len(samples)}</div>\n'
        '<div class="card"><h2>一、汇总</h2>'
        "<table><thead><tr><th>算法</th><th>平均 detect 耗时</th><th>总检出数</th>"
        "<th>PASS</th><th>FAIL</th></tr></thead>"
        f'<tbody>{summary_rows}</tbody>'
        f'<tfoot><tr><th colspan="5">判定一致率：{_escape(summary["consistency"])}</th></tr></tfoot>'
        "</table>"
        '<div class="note">判定一致率 = 两算法 is_pass 相同的样本数 / 样本总数；'
        "平均耗时为单帧 detect() 调用，已剔除各算法首帧预热（模型加载不计入），样本卡片仍显示原始单帧耗时。</div></div>\n"
        f'<div class="card"><h2>二、逐样本对比（{len(samples)} 张）</h2>'
        '<div class="note" style="margin-bottom:14px;">左栏 OpenCV，右栏 YOLO；'
        "标注图已 base64 内嵌，断网可看。</div>"
        f'{"".join(cards)}</div>\n'
        "</div>\n</body>\n</html>\n"
    )

    output.write_text(page, encoding="utf-8")
    logger.info("双算法对比 HTML 已生成：%s（样本 %d）", output, len(samples))
    return output
