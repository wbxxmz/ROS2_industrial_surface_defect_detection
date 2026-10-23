# 扩展 03：Web 演示服务（FastAPI + 拖拽上传页）

> 版本：V1.2 ｜ 所属模块：Web 演示服务 + 拖拽上传页（扩展计划总表 V1.2）
> 交付形态：无头服务 + 极简前端页面，不引入桌面 GUI，符合扩展原则 P5。
> 说明：阶段 9 §4.1 已把 FastAPI 列为后续接入点，本扩展把「检测 → 判定 → 标注 → 报告」能力以 HTTP 服务形态暴露，是「工具 → 产品」的标志。

## 1. 目标

- 提供一条可现场演示的 HTTP 链路：**拖拽 / 上传一张缺陷图 → 实时返回「标注图 + 判定结论 + 缺陷明细」→ 可导出 PDF / HTML 报告**。
- 复用既有 `pipeline/`、`report/`、`visualization/`，**不重写检测与报告逻辑**，把服务层做成薄封装。
- 面试 / 答辩场景下，无需命令行，打开浏览器即可「一键检测」，支持录屏展示。

## 2. 契约影响

- **不改动 `Detection` / `Verdict` / `Frame` / `Detector` / `InputSource` 任何既有契约**（符合扩展原则 P1）。
- 服务仅通过 `Detector.load(cfg)` × `detect(frame)`、`render_to_file`、`ReportBuilder.build` 组合既有能力；判定口径与 `judge` 共用同一份 `configs/*.yaml`（P2 配置驱动，P7 可复现）。
- 异常语义沿用：加载失败 `DetectorLoadError`、推理失败 `DetectorInferenceError`、输入非法 `ValueError`；服务层把上述异常统一转换为 4xx/5xx 与可读 JSON，不向客户端泄露堆栈（P8）。
- 新增依赖 `fastapi` + `uvicorn`，仅在启用 Web 路线时安装（与可选依赖 `torch` / `ultralytics` 的隔离策略一致），不阻塞 CLI 路线。

## 3. 核心方案

- 新增 `web/app.py`（应用工厂 + 生命周期）、`web/routes.py`（接口）、`web/static/`（极简前端页），保持「路由 / 业务」分离。
- **模型单例懒加载**：首个请求到达时才 `load(cfg)` 一次，之后复用；`app` 关闭时 `release()` 释放 GPU（幂等）。避免每次请求重新加载权重带来的秒级延迟。
- 接口设计（先最小集，后续不外扩）：
  1. `GET  /` —— 返回内嵌静态页（拖拽上传区）。
  2. `POST /api/detect` —— `multipart/form-data` 接收图片，返回 JSON：`is_pass`、`reason`、`max_severity`、`detections[]`（class_name / confidence / bbox / severity）、`annotated_url`（标注图）、`attention_url`（EigenCAM 热力图，依赖 YOLO 时可选）。
  3. `GET  /api/images/{name}` —— 返回标注图 / 热力图（由 `/api/detect` 生成后落盘到会话目录）。
  4. `GET  /api/report?format=pdf|html` —— 只对「当前样本」生成并返回报告文件下载（不做跨请求聚合）。
- **输入安全边界**（对应总表「五、风险」中的 Web 安全项）：限制上传大小（如 ≤ 10 MB）、仅接受图片扩展名、解码失败返回 422；临时上传 / 产物放入可清理目录，避免把本地推理环境暴露成不可控外部服务。
- 标注图复用 `visualization/renderer.py` 的 `render_to_file`；热力图复用 `visualization/attention.py`（YOLO）/ `heatmap.py`（通用）；报告复用 `report/builder.py`。
- 前端：纯静态 HTML + fetch，无构建链；拖拽区 + 结果区（标注图 / 热力图并排）+ 结论徽标 + 报告下载按钮，样式与既有 HTML 报告 CSS 保持一致。
- 复现命令：`uvicorn web.app:create_app --factory --host 127.0.0.1 --port 8000`，浏览器打开 `http://127.0.0.1:8000`。

### 核心方案补充约束（实现落地）

以下为实现时追加的硬约束，代码已按此落地：

1. **并发**：`app.state.detector` 单例；`load` 用 `threading.Lock` **双重检查**，防止首个并发请求重复加载；每次 `detect`（含分级 / 判定 / 渲染）在同一把锁内**串行**；端点统一写成 `async def` + `await asyncio.to_thread(...)`，不阻塞事件循环。
2. **异常映射**：`SourceOpenError` → 422、`DetectorInferenceError` → 500、`ValueError` → 400、其余未预期异常 → 500 且只记服务端日志；响应体统一 `{"error": "...", "detail": "..."}`，**禁止返回 traceback**。
3. **依赖**：`requirements.txt` 增加 `fastapi`、`uvicorn`、`python-multipart`（`fastapi.testclient` 另需 `httpx`）。
4. **报告接口简化**：`/api/report` 只针对「当前请求的这一张」生成，`batch=[当前样本]`；**不做跨请求会话聚合**，传旧 `sample_id` 直接 409。
5. **上传边界**：扩展名白名单 `.jpg/.jpeg/.png/.bmp`，单文件 ≤ 10 MB；**解码失败统一返回 422**（不与 400 混用）。
6. **产物访问**：`/api/images/{name}` 只返回本服务本次生成的产物，文件名做白名单校验，禁止路径穿越。
7. **测试**：`tests/test_web.py` 用 `TestClient` 覆盖正常上传、坏文件、超大文件、错误扩展名、空文件、产物白名单、单样本报告，**不启动真实端口**。

## 4. 验收标准

- `POST /api/detect` 上传 `test_picture/inclusion_174.jpg`，返回 JSON 中 `detections` 与 YOLO 单机 `detect()` 口径一致（取值 / 置信度 / bbox 像素坐标逐字段一致）。
- 浏览器页面拖拽同一张图，能看到**标注图 + 结论（PASS/FAIL）+ 缺陷明细**；YOLO 路线下同步展示 EigenCAM 注意力热力图。
- `GET /api/report?format=html` 能下载自包含 HTML 报告，断网双击可开，样本数 / 判定与终端日志一致。
- 上传非法文件（非图片 / 超大小 / 空文件）返回 4xx 结构化错误，不崩溃、不泄露堆栈。
- 未加载模型时首个请求触发 `load`，进程关闭调用 `release`（日志可见「已释放 YOLO 模型与显存缓存」，二次启动可复现）。

## 5. 风险说明

- **并发与单例模型**：单例模型非线程安全时，需用锁或进程内串行化推理，避免并发请求竞争；演示场景单并发即可，产线化再评估多 worker / 队列。
- **安全边界**：本服务仅限本地 / 内网演示，需加输入大小限制与白名单扩展名；不直接暴露到公网，避免把推理环境变成可被滥用的外部服务。
- **可选依赖隔离**：`fastapi` / `uvicorn` 仅在启用 Web 路线时导入与安装，CLI 路线零新增依赖。