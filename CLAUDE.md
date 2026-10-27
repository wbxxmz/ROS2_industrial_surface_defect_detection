# CLAUDE.md

> 给新对话的接手人（AI 或人）：先读本文件，再动手。这里只写“不看代码就会踩坑”的信息，细节以代码和 `docs/` 为准。

## 1. 项目定位

Python 工业产品表面缺陷检测平台，用于求职作品与答辩演示：

- 双算法：OpenCV 传统算法 + YOLOv8m（`best.pt`），可单独跑也可对比跑；
- 配置驱动、无头架构：CLI 批处理 + FastAPI Web 演示，不建 GUI 窗口；
- 产物：标注图 jpg、PDF / Excel / 自包含 HTML 报告；
- 另有一套 ROS2 包（`src/`），日常不改、不演示。

## 2. 硬约束（改动前必读）

- **契约不可破坏**：`Detection`（`class_id` / `class_name` / `bbox` 像素 `(x, y, w, h)` / `area_px` / `area_ratio` / `confidence` / `mask` / `severity`）、`Frame`、`Verdict`、`Detector` 抽象基类、`InputSource`；`Detection` / `Frame` 在 `core/types.py`，`Verdict` 在 `core/verdict.py`，`Detector` / `InputSource` 在 `core/detector.py` / `core/input_source.py`。
- **类别映射固定**：`scratch:0` / `dent:1` / `stain:2` / `notch:3`（NEU-DET 6 类合并为 4 类），后续所有模块复用，不得另建映射。
- **阈值全部走 `load(cfg)` / YAML**，禁止在代码里硬编码新阈值；新增参数必须同步写进 `configs/`。
- **无头**：不引入 PyQt 等 GUI，只落盘。
- **异常语义固定**：`ValueError`、`DetectorLoadError`、`DetectorInferenceError`、`SourceOpenError`、`InputSourceError`、`ProjectError`、`OSError`。Web 错误体统一 `{"error": "...", "detail": "..."}`，禁止返回 traceback。
- `Detection.severity` 只能是 `0/1/2` 或 `None`；可视化文字只用英文（中文会被 cv2 画成乱码）。
- 空帧 / 零尺寸输入 → 抛 `ValueError` 或返回空列表；面积 < 10px 的轮廓忽略；OpenCV 检测器丢弃 bbox 面积 > 全图 15% 的结果（YOLO 不适用）。
- **不得为了让验收好看而调阈值或伪造数据**。实测与预期不符时，如实报告并说明原因。

## 3. 目录职责

| 路径 | 职责 |
| --- | --- |
| `core/types.py` | `Frame` / `Detection` 等契约 dataclass（`Verdict` 在 `core/verdict.py`） |
| `inputs/` | 图片/目录/视频/相机输入源 |
| `detectors/` | `opencv_detector.py`、`yolo_detector.py`、`dummy_detector.py` |
| `pipeline/pipeline.py` | 主流水线：detect → grade → judge → visualize → report |
| `severity/` `judge/` | 严重度分级、判定规则（参数来自 cfg） |
| `report/` | `builder.py` + `exporters/`（pdf / xlsx / html / comparison_html） |
| `visualization/` | 标注图、密度热力图、EigenCAM 注意力图 |
| `web/` | FastAPI：`app.py`（工厂 + 异常映射）、`routes.py`、`static/index.html` |
| `scripts/` | `benchmark.py`、`compare_algorithms.py`、`compare_html.py` |
| `configs/` | 4 份配置，见下节 |
| `tests/` | 测试用例 |
| `docs/` | 实现文档、需求扩展、模型对比、演示脚本、README 配图、`进度日志.md`（跨对话状态存档） |
| `run_demo.sh` | 一键演示脚本（自检依赖 → 批处理） |

## 4. 常用命令

```bash
# 一键演示（自检依赖 → 批处理）
./run_demo.sh opencv          # OpenCV + PDF
./run_demo.sh yolo            # YOLO + PDF
./run_demo.sh benchmark       # 批量压测

# 批量报告（HTML 自包含）
python3 main.py --config configs/html_demo.yaml        # OpenCV HTML
python3 main.py --config configs/html_yolo_demo.yaml   # YOLO HTML + 注意力图

# 双算法三栏对比页
python3 scripts/compare_html.py

# Web 演示（模型懒加载，默认跑 YOLO）
uvicorn web.app:create_app --factory --host 127.0.0.1 --port 8000
DEFECT_WEB_CONFIG=configs/html_demo.yaml uvicorn web.app:create_app --factory   # 换 OpenCV

# 测试
python3 -m pytest -q
```

产物全部在 `reports/`（已被 `.gitignore` 忽略）：`quality_report*.pdf/html`、`algorithm_comparison.html`、`benchmark_result.json`、`visual/<图名>.jpg`、`visual/compare/*`、`web/<request_id>_*.{jpg,html,pdf}`。HTML 都是 base64 内嵌，`xdg-open` 直接打开即可，无需服务器。

## 5. 配置对应关系

| 配置 | 算法 | 输出 |
| --- | --- | --- |
| `configs/demo.yaml` | OpenCV | `quality_report.pdf` |
| `configs/yolo_demo.yaml` | YOLOv8m | `quality_report_yolo.pdf` |
| `configs/html_demo.yaml` | OpenCV | `quality_report.html` |
| `configs/html_yolo_demo.yaml` | YOLOv8m + `attention: true` | `quality_report_yolo.html` |

判定/分级参数：`max_count`、`max_area_ratio`、`min_confidence`、`min_area_px`、`max_severity`、`severity.enabled`。

## 6. 协作习惯（务必遵守）

- 全程中文回复；代码标识符、commit message 用英文。
- **先计划、后动手**：涉及新增模块或改动契约时，先在 `docs/需求扩展/` 写轻量文档（固定四块：目标 / 契约影响 / 核心方案 / 验收标准），确认后再写代码。
- **先只读核对，再改**：涉及既有代码时，先贴出文件路径 + 行号 + 原文，不凭记忆下结论。
- **最小改动**：只做被要求的事，不顺手重构、不提前抽象。
- **冲突先停**：指令与验收标准冲突，或任务要求与既有规则冲突时，先停下说明冲突点并提问，不要自行选一边执行。
- **测试纪律**：先跑 `pytest -q`；不绿立即停下贴失败详情，**不许放宽断言去迁就实现**。新增用例时只新增，不改已有用例。
- **提交纪律**：基线与补测试分开提交；未经明确指令不 `git push`；已存在的 tag 不用 `-f` 覆盖。
- **报告纪律**：说清“做了什么 / 没做什么 / 已知限制”；负面结果（漏检、效果不明显、无法下载）照实说。
- **状态自更新（无需用户提醒）**：每完成一项有实质产出的任务（代码 / 数据 / 实验 / 结论 / 决策），主动刷新 `docs/进度日志.md`——§1 当前快照**覆盖重写**（时间、焦点、工作区状态、能力边界、最近验证、阻塞），§2 待办按实际增删勾选，§3 时间线**只追加**（日期 + 做了什么 + 结论 + 证据路径或 commit）。纯问答、无仓内产出、无新结论的对话不写；写结论必须附文件行号、产物路径或实测数据，禁止凭记忆补记。
- 临时脚本、调试图不要放在仓库根目录；中间产物放 `reports/` 或 `/tmp`。

## 7. 已知坑位

- **EigenCAM**：ultralytics 在推理时会 `deepcopy` 模型，hook 必须挂在 `inference_model` 上，且要 `try/finally` 移除；默认目标层是 backbone stage-4 C2f（idx 6，40×40），不要换回 SPPF（idx 9）。
- **灰度图**：`test_picture/` 15 张全是纯灰度（S=0），`stain` 分支永不触发，这是数据问题不是链路问题。
- **彩色图**：`stain` 置信度 = Lab 色距 / 441，普通彩色污渍约 0.2，被 `min_confidence=0.45` 过滤；调低阈值可检出，但同时会增加误检。
- **YOLO 泛化**：只在 NEU-DET 域内可靠；任意网图属域外输入，输出无意义。
- **Web 测试**：`TestClient` 退出 `lifespan` 会 `release()` 并把 `app.state.detector` 置回 `None`，断言必须写在 `with` 块内。
- **目录输入**：目录源 `read()` 失败会卡在同一游标，批量场景要逐文件处理才能跳过坏图。(已复现)
- **HTML**：所有字段先 `html.escape()`；图片缺失渲染灰底占位；必须 base64 内嵌，保证断网可开。
- **历史文档是快照**：`docs/实现文档/阶段N_*.md` 里的目录树/结构记录的是当时状态，不随代码变化回改。例：阶段 1 文档仍列 `ui/`（该包已在 860643b 删除），属正常，不要为“看起来一致”去改历史文档。
- **根目录脚本**：`train_yolo.py`、`test_density_heatmap.py`、`test_eigencam.py`
  已入库；`visualize.py`、`visualize_yolo.py` 已被 `.gitignore` 忽略。
- **samples_color/**：本地彩色样本目录，已在 `.gitignore:49` 忽略，不入库、不参与演示产物。
- **彩色域外评估（2026-10-27）**：FabricDefect（MIT，24 张彩色织物，`samples_color/fabric_defect/`）实测 OpenCV 143 框（`stain=0`、GT 覆盖最高 2.4%）、YOLO 29 框（20 框中心在 GT 外、9 张零检出）；结论是“彩色可跑通但不可靠”，主 demo 仍以灰度 NEU-DET 为准，彩色不作为能力声明。详见 `docs/模型对比/彩色域外评估_FabricDefect.md`。

## 8. 新增功能标准流程

1. 只读核对现状（代码 + 配置 + 测试）；
2. 写/更新 `docs/需求扩展/` 文档并等确认；
3. 最小实现 + 补测试；
4. `python3 -m pytest -q` 全绿；
5. 汇报产物路径与验收数据；
6. 用户确认后再 commit（不 push）。
