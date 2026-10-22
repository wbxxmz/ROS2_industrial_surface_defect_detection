# ROS2_industrial_surface_defect_detection

> 基于 ROS2 + Python 的工业产品表面缺陷检测平台，面向产线质检场景，支持**图片 / 视频 / 摄像头**三类输入，提供 **OpenCV 传统算法** 与 **YOLO 深度学习** 双算法路线。

## 项目简介

本项目以 ROS2 节点架构组织检测流水线，把"输入采集 → 预处理 → 检测 → 后处理 → 结果输出"拆分为职责单一的节点与可替换组件，便于按产线需求切换输入源与检测算法，而无需改动主流程代码。

- **多源输入**：图片、视频文件、摄像头，统一产出帧对象。
- **双算法路线**：OpenCV 传统视觉（阈值 / 边缘 / 形态学等）与 YOLO 模型检测，二者通过同一抽象接口对外服务。
- **产线可用**：检测结果进入可视化与质检报告环节，形成"检测 → 判定 → 标注图 → PDF/Excel 报告"闭环。
- **无头架构**：不依赖 GUI，全部结果落盘为图片与报告，可直接对接后续 Web / ROS2 节点。

## 📊 双算法基线对比 (Benchmark)

在 NEU-DET 验证集的 **30 张真实工业缺陷样本**（按缺陷类别分层抽样，每类 5 张）上，用同一套判定规则（`judge.min_confidence=0.25`、`max_count=0`、`max_area_ratio=0.01`）分别运行两种检测器：

| 指标 | OpenCV 传统算法 | YOLOv8m (best.pt) | 双算法并集 |
| --- | --- | --- | --- |
| 检出率（至少检出 1 条有效缺陷） | 40.0%（12/30） | **100.0%（30/30）** | **100.0%（30/30）** |
| mAP50（360 张验证集） | — | **0.7693** | — |
| 平均单帧耗时 | **3.7 ms**（CPU） | 15.0 ms（GPU，中位 15.3 ms） | — |
| 典型误检倾向 | 复杂纹理易产生伪候选 | 低置信度下偶有重叠大框 | 两类漏检互相补齐 |

**结论**：最终采用 **YOLOv8m + 640 分辨率**，在 30 张验证样本上检出率达到 **100%**，mAP50 为 **0.7693**，且已验证 **YOLO 完全覆盖了 OpenCV 的漏检**（`OpenCV FAIL → YOLO PASS` 为 0）。OpenCV 仍以毫秒级速度承担高速初筛，YOLO 负责把漏检补齐。

### 工程权衡（Trade-off）

为了达到 **100% 的极限召回率**，本项目把置信度阈值下调到 `0.10`，代价是同一张图会输出部分互相重叠的大框（例如 `test_picture/crazing_98.jpg` 上出现 5 个 scratch 框）。在真实工业场景中，可以调高阈值并配合更严格的 NMS 策略（提高 `iou_threshold` 的严格程度、按类别做重叠抑制）来减少冗余框；如果产线节拍优先，也可以直接降级到 YOLOv8n/s，用部分微小缺陷的召回换取更快的 CPU 推理速度。

数据口径：NEU-DET 验证集样本均为缺陷件，因此"检出率"指至少检出 1 条有效缺陷的比例；耗时仅统计单帧 `detect()` 调用，已排除模型加载与首次预热；30 张为单次分层抽样，结论为方向性参考。完整汇总、判定一致率与逐图明细见 [comparison_report.md](comparison_report.md)，复现命令为 `python3 scripts/compare_algorithms.py --limit 30`。

## 🏭 工业落地思考

为了追求极致的召回率，本项目最终采用了 **YOLOv8m + 640 分辨率** 的配置。但在真实的工业产线中，如果面临极高的节拍要求，可通过配置化接口无缝降级到 **YOLOv8n/s**，牺牲部分微小缺陷召回率以换取 **40+ FPS** 的极速检测。

## 架构亮点

| 亮点 | 说明 |
| --- | --- |
| **契约先行** | 先冻结接口契约再写实现，各阶段严格按契约对齐，避免"边写边改签名"。 |
| **统一抽象接口** | `Detector`（`load / detect / name / version / release`）与 `InputSource`（`open / read / close`）是算法与输入源唯一对外的契约面。 |
| **统一数据结构** | `Frame` 与 `Detection`（含 `severity` 可选字段）字段名、类型、单位均已冻结。 |
| **自定义异常体系** | `DetectorLoadError`、`DetectorInferenceError`、`SourceOpenError`、`InputSourceError` 统一继承 `ProjectError`。 |
| **配置驱动** | 检测器参数、判定规则、分级权重、报告格式全部来自 YAML，不在代码里写死阈值。 |

## 环境要求

- **操作系统**：Linux（Ubuntu 22.04 及兼容发行版）
- **Python**：≥ 3.10
- **必装依赖**：见 `requirements.txt`（NumPy、OpenCV、PyYAML、reportlab、openpyxl、matplotlib、psutil）
- **可选 CUDA 环境**：NVIDIA 驱动 + CUDA 版 PyTorch（`torch`）与 `ultralytics`，仅 YOLO 路线需要；无 GPU 时 OpenCV 路线可正常使用

## 安装步骤

```bash
# 1. 克隆仓库
git clone <repo-url>
cd ROS2_industrial_surface_defect_detection

# 2. 安装必装依赖
pip install -r requirements.txt

# 3.（可选）YOLO 路线：安装 GPU 版 PyTorch 与 ultralytics
pip install torch ultralytics        # 有 CUDA 时按官网选择对应 cu126 轮子

# 4. 一键验证
bash run_demo.sh
```

## 三种运行方式

### 1）一键启动（推荐）

```bash
./run_demo.sh              # 默认 OpenCV 路线
./run_demo.sh opencv       # OpenCV 传统算法
./run_demo.sh yolo         # YOLOv8m（需 best.pt）
./run_demo.sh benchmark    # 批量压测（30 张，输出 JSON 指标）
```

脚本会先自检 Python 版本、核心依赖、样例图片；缺失时打印中文提示与安装命令并返回非零码，不会抛出裸 traceback。

### 2）OpenCV 路线

```bash
python3 main.py --config configs/demo.yaml
# 产出：reports/quality_report.pdf + reports/visual/*.jpg
```

### 3）YOLO 路线

```bash
python3 main.py --config configs/yolo_demo.yaml
# 产出：reports/quality_report_yolo.pdf + reports/visual/*.jpg
```

### 4）批量压测

```bash
python3 scripts/benchmark.py --config configs/yolo_demo.yaml \
    --input tests/data/stress_images --limit 30
# 产出：reports/benchmark_result.json（耗时 / FPS / 峰值内存 / 错误清单）
```

> 压测素材 `tests/data/` 为本地生成、不入库（见 `.gitignore`）。可自行准备任意图片目录；脚本对损坏图片与空目录均做了异常隔离，不会中断整批。

## 模型权重获取方式

`best.pt`（约 52 MB）体积较大，**不随仓库分发**（`.gitignore` 已忽略 `*.pt`）。两种获取方式任选其一：

**方式 A：训练脚本复现**

```bash
python3 train_yolo.py     # NEU-DET 微调，结束后自动把 best.pt 复制到项目根目录
```

需要 `shapes/NEU-DET/` 数据集；训练在 GPU 上约需数十分钟到数小时（取决于显卡与轮数）。

**方式 B：外部下载**

从项目 Release 附件下载 `best.pt` 并放到**项目根目录**：

```
https://github.com/wbxxmz/ROS2_industrial_surface_defect_detection/releases
```

若 Release 尚未上传权重，请使用方式 A 复现。

## 性能基线

测试条件：30 张 200×200 工业缺陷图；RTX 4060 Laptop 8GB（YOLO）/ 同机 CPU（OpenCV）；`conf_threshold=0.10`、`imgsz=640`；计时仅统计单帧 `detect()`，已排除模型加载与首帧预热。

| 指标 | OpenCV | YOLOv8m |
| --- | --- | --- |
| 平均检测耗时 | **3.7 ms** | **15.0 ms** |
| 检测 FPS | ≈ 270 | **66.6** |
| 端到端 FPS（含判定/渲染） | — | **60.9** |
| 峰值内存（RSS） | 48 MB | 1413 MB |
| 30 张总耗时 | — | 1.72 s |

复现命令：`./run_demo.sh benchmark`，或 `python3 scripts/benchmark.py --config configs/yolo_demo.yaml --input tests/data/stress_images --limit 30`。

## 目录结构

```
ROS2_industrial_surface_defect_detection/
├─ core/            # 接口与数据结构：Frame / Detection / Detector / InputSource / Verdict
├─ inputs/          # 输入源实现：图片 / 视频 / 摄像头 + 工厂
├─ detectors/       # 检测器实现：OpenCV / YOLO（含 DummyDetector 占位）
├─ pipeline/        # 流水线编排：采集 → 检测 → 分级 → 判定 → 可视化
├─ judge/           # 判定规则（is_pass / reason / max_severity）
├─ severity/        # 缺陷严重程度分级
├─ visualization/   # 无头标注图渲染（BGR 色表 / PASS-FAIL 横幅）
├─ report/          # 质检报告：ReportBuilder + PDF / Excel 导出
├─ configs/         # 配置样例：demo.yaml / yolo_demo.yaml
├─ scripts/         # benchmark.py / compare_algorithms.py
├─ tests/           # 单元测试与集成用例
├─ docs/            # 各阶段实现文档
├─ reports/         # 产物目录（报告与标注图，不入库）
├─ utils/           # 日志、配置加载、自定义异常
├─ run_demo.sh      # 一键启动脚本
├─ main.py          # 程序入口
└─ requirements.txt # Python 依赖清单
```

## 已知限制

- **无 CUDA 时 YOLO 不会自动回退 CPU**：`configs/yolo_demo.yaml` 中 `device: cuda`，若当前环境 `torch.cuda.is_available()` 为 `False`，会明确抛出 `DetectorLoadError`（报错信息含"不静默回退"）并返回非零退出码。这是**刻意设计**，不是 Bug——避免用户误以为在用 GPU 而实际跑在 CPU 上。无 GPU 时请改用 `configs/demo.yaml`，或显式把 `device` 改为 `cpu`。
- **模型权重不入库**：见上文「模型权重获取方式」。
- **报告字体**：PDF 中文依赖系统字体（Arphic UMing），若系统缺失会回退到 reportlab 内置字体并可能显示异常。
- **压测素材不入库**：`tests/data/` 需本地准备。

## 错误处理约定

`main.py` 对错误统一转换为可读提示并返回非零退出码：

| 异常 | 含义 | 退出码 |
| --- | --- | --- |
| `DetectorLoadError` | 检测器加载失败（权重缺失 / 设备不可用 / 参数非法） | 1 |
| `DetectorInferenceError` | 推理失败（含 CUDA OOM） | 1 |
| `SourceOpenError` | 输入源打开失败（目录为空 / 图片损坏 / 目录不存在） | 1 |
| `InputSourceError` | 读取中断流（解码失败） | 1 |
| `ValueError` | 配置错误（如未知 `detector.type`） | 1 |
| `FileNotFoundError` | 配置文件不存在 | 1 |

## 当前进度

阶段 0–9 全部完成：接口契约冻结、输入源、OpenCV/YOLO 双检测器、判定与严重度分级、无头可视化、PDF/Excel 质检报告、双算法对比、批量压测与性能基线、一键部署交付。

## License

本项目采用 [MIT License](LICENSE)。
