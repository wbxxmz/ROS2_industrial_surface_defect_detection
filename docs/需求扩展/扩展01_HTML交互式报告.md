# 扩展 01：HTML 交互式报告

> 版本：V1.1 ｜ 所属模块：HTML 交互式质检报告（扩展计划总表 V1.1）
> 交付形态：轻量扩展，不改变既有契约，新增一个导出格式。

## 1. 目标

- 在 PDF / Excel 之外新增「浏览器双击即开」的**自包含 HTML 质检报告**，作为可直链分享、可挂作品集的展示成果。
- 复用 `ReportBuilder` 已组装好的报告数据，不重算统计口径，保证 HTML 与 PDF / Excel 的样本数、判定、明细**逐字段一致**。

## 2. 契约影响

- 不改动 `Detection` / `Verdict` / `Frame` 任何字段（符合扩展原则 P1）。
- 对外入口保持 `ReportBuilder.build(batch, cfg) -> Path` 不变；仅把 `SUPPORTED_FORMATS` 从 `("pdf", "xlsx")` 扩展为 `("pdf", "xlsx", "html")`，并在 `build()` 的路由里新增 `html` 分支（与现有 pdf/xlsx 分支同构）。
- `ReportBuilder._assemble()` 产出的 `data`（`summary` / `samples` / `details` / `class_names` / `overall_max_severity` …）作为 HTML 导出器的**唯一输入**，不新增中间结构。
- `image_path` 指向的图片缺失时，渲染**灰底占位块 +「图片缺失」**文字，不抛异常（与 PDF 导出器的占位语义一致）。
- 异常语义沿用 builder §4.4：`cfg.format` 非支持值抛 `ValueError`；写盘失败（无权限 / 磁盘满）由底层 `OSError` 原样上抛。

## 3. 核心方案

- 新增 `report/exporters/html_exporter.py`，实现 `export_html(data, output)`。
- **模板与数据分离**：优先用原生 `str.format` / `string.Template` 渲染，避免引入 Jinja2 依赖；结构简单，无需大模板引擎（若后续字段复杂再评估引入 Jinja2，届时在 `requirements.txt` 登记）。
- **自包含**：统计图与标注图读取本地文件后经 **base64** 内嵌到 `<img>`；CSS 内联 `<style>`；概览表与明细表直接用 `data` 渲染为 HTML `<table>`。
- 统计图复用 `report/charts.py` 的 `class_distribution_chart()` 产出 PNG（`matplotlib Agg` 无头后端），再 base64 内嵌；标注图取 `data['samples'][].image_path` 指向的图片。
- **统一转义**：所有文本字段（标题 / 样本名 / `reason` / 类别名 / 参数等）渲染前统一经 `html.escape()`，防止 `> < &` 等字符破坏页面结构。
- **图片缺失降级**：`image_path` 为空或文件不存在时，渲染灰底占位块 +「图片缺失」，仅记 warning，不抛异常、不中断导出。
- 接入：`report/builder.py` 的 `SUPPORTED_FORMATS` 增加 `"html"`；`build()` 增加 `html` 分支调用 `export_html(data, output)`；默认 `output_path` 设为 `reports/quality_report.html`（由 `configs/*.yaml` 的 `report` 段配置 `format: html`）。
- 复现命令：`python3 main.py --config configs/demo.yaml`（`report.format: html`）。

## 4. 验收标准

- 一条命令在 `reports/quality_report.html` 生成**单个 HTML 文件**。
- 浏览器双击打开：能看到标注图、概览表、缺陷明细表、类别统计图；页面**无外链资源**（断网可打开）。
- HTML 内的样本数、合格 / 不合格数、缺陷明细行数、有效缺陷计数，与终端日志及 PDF / Excel 报告逐字段一致。
- 含 `> < &` 的 `reason` 文本在页面中**原样正常显示**（转义生效），不破坏页面结构。