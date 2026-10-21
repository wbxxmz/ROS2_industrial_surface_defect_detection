# ROS2_industrial_surface_defect_detection

> 基于 ROS2 + Python 的工业产品表面缺陷检测平台，面向产线质检场景，支持**图片 / 视频 / 摄像头**三类输入，提供 **OpenCV 传统算法** 与 **YOLO 深度学习** 双算法路线。

## 项目简介

本项目以 ROS2 节点架构组织检测流水线，把"输入采集 → 预处理 → 检测 → 后处理 → 结果输出"拆分为职责单一的节点与可替换组件，便于按产线需求切换输入源与检测算法，而无需改动主流程代码。

- **多源输入**：图片、视频文件、摄像头，统一产出帧对象。
- **双算法路线**：OpenCV 传统视觉（阈值 / 边缘 / 形态学等）与 YOLO 模型检测，二者通过同一抽象接口对外服务。
- **产线可用**：检测结果可进入可视化与质检报告环节，形成闭环。

## 📊 双算法基线对比 (Benchmark)

在 NEU-DET 验证集的 **30 张真实工业缺陷样本**（按缺陷类别分层抽样，每类 5 张）上，用同一套判定规则（`judge.min_confidence=0.25`、`max_count=0`、`max_area_ratio=0.01`）分别运行两种检测器：

| 指标 | OpenCV 传统算法 | YOLO (best.pt) | 双算法并集 |
| --- | --- | --- | --- |
| 检出率（至少检出 1 条有效缺陷） | 40.0%（12/30） | 86.7%（26/30） | **90.0%（27/30）** |
| 平均单帧耗时 | **3.7 ms** | 23.7 ms（中位） | — |
| 典型误检倾向 | 复杂纹理易产生伪候选 | 低对比度小缺陷偶有漏检 | 两类漏检互相补齐 |

**结论**：双算法组合有效互补——OpenCV 以毫秒级速度承担产线高速初筛，YOLO 在此基础上大幅降低漏检率；两者并集使检出率从 40% 提升到 90%，同时保留了传统算法的低延迟优势。

数据口径：NEU-DET 验证集样本均为缺陷件，因此"检出率"指至少检出 1 条有效缺陷的比例；耗时仅统计单帧 `detect()` 调用，已排除模型加载与首次预热；30 张为单次分层抽样，结论为方向性参考。完整汇总、判定一致率与逐图明细见 [comparison_report.md](comparison_report.md)，复现命令为 `python3 scripts/compare_algorithms.py --limit 30`。

## 架构亮点

| 亮点 | 说明 |
| --- | --- |
| **契约先行** | 先冻结接口契约再写实现，各阶段严格按契约对齐，避免"边写边改签名"。 |
| **统一抽象接口** | `Detector`（`load / detect / name / version / release`）与 `InputSource`（`open / read / close`）两大抽象基类，是算法与输入源唯一对外的契约面；空壳方法体只抛 `NotImplementedError`，签名不得改动。 |
| **统一数据结构** | `Frame`（`image / width / height / frame_index / timestamp`）与 `Detection`（`class_id / class_name / bbox / area_px / area_ratio / confidence / mask / severity`），字段名、类型、单位均已冻结，后续阶段只允许新增可选字段。 |
| **自定义异常体系** | `DetectorLoadError`、`DetectorInferenceError`、`SourceOpenError`、`InputSourceError` 统一继承 `ProjectError`，上层只需捕获 `ProjectError` 即可统一处理运行期错误。 |

## 当前进度

- ✅ **阶段 0**：需求与总体设计、接口契约（`Frame` / `Detection` / `Detector` / `InputSource`）冻结完成。
- ✅ **阶段 1**：基础框架搭建完成——抽象基类、数据结构、异常体系、流水线骨架（直通占位）、配置加载与日志系统落地。
- ✅ **冒烟测试通过**：端到端流程"加载占位帧 → 检测（空实现）→ 输出空结果"可跑通。
- ⏳ 后续阶段：输入源实现 → 检测器实现 → 可视化 → 配置 → 报告 → 批处理 → CLI/UI → 测试验收。

## 目录结构

```
ROS2_industrial_surface_defect_detection/
├─ core/            # 接口与数据结构：Frame / Detection / Detector / InputSource
├─ inputs/          # 输入源实现：图片 / 视频 / 摄像头
├─ detectors/       # 检测器实现：OpenCV / YOLO（含 DummyDetector 占位）
├─ pipeline/        # 流水线编排：采集 → 预处理 → 检测 → 后处理 → 输出
├─ ui/              # 结果可视化与判定
├─ report/          # 质检报告生成
├─ configs/         # 配置样例（如 demo.yaml）
├─ tests/           # 单元测试与冒烟测试
├─ utils/           # 日志、配置加载、自定义异常
├─ main.py          # 程序入口
└─ requirements.txt # Python 依赖清单
```

## 环境要求

- Ubuntu 22.04（或兼容发行版）+ ROS2（Humble 及以上）
- Python 3.10
- 依赖：OpenCV、NumPy、PyYAML、pytest（见 `requirements.txt`）

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 编译工作空间（ROS2）

```bash
colcon build
source install/setup.bash
```

### 2. 运行流水线骨架

```bash
python3 main.py --config demo.yaml
```

正常输出示例：

```
[INFO] 已加载配置：.../configs/demo.yaml
[INFO] 流水线完成：帧 1×1，检测结果 0 条
[INFO] 运行完成，共 0 条检测结果
```

### 3. 运行测试

```bash
python3 -m pytest tests/ -v
```

## 错误处理约定

`main.py` 对三类错误统一转换为可读提示并返回非零退出码：

| 异常 | 含义 | 退出码 |
| --- | --- | --- |
| `ProjectError` | 运行错误（检测器加载 / 推理 / 输入源打开 / 读取异常） | 1 |
| `ValueError` | 配置错误（如未知的 `detector.type`） | 1 |
| `FileNotFoundError` | 配置文件不存在 | 1 |
