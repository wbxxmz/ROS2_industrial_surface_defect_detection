# 阶段 4：YOLO 模型集成 · 实现技术文档

## 1. 阶段目标

集成 YOLO 系列深度学习检测器，与 OpenCV 路线并列，实现真正的"双算法可切换"，并提升对复杂缺陷的泛化能力。

## 2. 前置条件与依赖

- 依赖阶段 0 的 `Detector` 接口、阶段 2 的帧输入、阶段 3 的对照结论。
- 依赖 PyTorch、YOLO 推理/训练库（如 ultralytics）。
- 需准备缺陷标注数据（阶段 0/3 的样本基础上扩展标注）。
- 依赖阶段 3 §4.1 的四类缺陷 `class_id`/`class_name` 映射表（**唯一权威映射，本阶段只复用、不重定义**，见 §4.1）。

## 3. 工作分解（WBS）

1. 数据标注与格式整理（如 YOLO 标注格式，含四类缺陷）。
2. 数据划分：训练/验证/测试集，处理类别不平衡。
3. 模型训练：选 baseline（YOLOv5/v8/v11），调关键超参。
4. 推理封装：将模型推理输出转换为统一的 `Detection` 结构（§4.2 契约转换）。
5. 实现 `YoloDetector(Detector)`：`load` 加载权重、`detect` 推理、`release` 释放显存（§5）。
6. 定义 `YoloDetector` 的配置参数与校验（`model_path`/`conf_threshold`/`iou_threshold`/`device`，§4.4）。
7. 接入主流程，与 OpenCV 检测器可配置切换。
8. 评测：mAP / 召回 / 精确率，与阶段 3 对照。

## 4. 关键技术方案

### 4.1 标注与类别

- 类别固定四类 + 可选背景（正常类），类别 id 与阶段 0/3 统一。
- **类别映射表复用阶段 3 §4.1**（`0=scratch、1=dent、2=stain、3=notch`）。本阶段不得重排、不得另起映射；训练标注的类别 id 必须与该表逐项一致，否则报告/统计错乱（§8 特殊风险点名）。
- 标注形式：边界框（目标检测）为基础；如需更准面积，后续可扩展为实例分割模型（改变 `area_px`/`mask` 口径，见 §4.2）。

### 4.2 推理封装与契约转换

模型输出（以 ultralytics `results.boxes` 为例，张量 `[N, 6]`，列序 `[x1, y1, x2, y2, conf, cls]`，像素坐标 xyxy）逐条转换为符合阶段 0 §5.4.1 的 `Detection`，规则如下：

| Detection 字段 | 转换规则 | 单位 / 口径 |
| --- | --- | --- |
| `class_id` | `int(cls)`，必须落在阶段 3 §4.1 映射表内；越界（∉ `{0,1,2,3}`）的结果丢弃并记 `warning` 日志，不中断整帧 | `int`，不可 `None` |
| `class_name` | `CLASS_MAP[class_id]` | `str`，不可 `None` |
| `bbox` | 由 `(x1,y1,x2,y2)` 转左上原点 `(x, y, w, h) = (x1, y1, x2-x1, y2-y1)`，四值 `int`，并 clamp 到图像边界 `[0, W]×[0, H]` | 像素、`int`、非归一化 |
| `confidence` | `float(conf)`，`clip(0,1)`；`< conf_threshold` 的框在转换前已被过滤 | `float` 0~1，恒非 `None` |
| `area_px` | 框级检测无掩码，以 bbox 面积近似：`area_px = w*h`（⚠ 与阶段 3「实际像素面积」口径有差，见 §8 特殊风险） | `int`，不可 `None` |
| `area_ratio` | `area_px / (frame.width * frame.height)`；空帧已在 `detect` 入口抛 `ValueError`，故 `W*H > 0` 恒成立，无除零 | `float` 0~1，不可 `None` |
| `mask` | 框级检测默认 `None`；若扩展为实例分割，填 `uint8` 单通道 `0/255` 掩码且 `area_px = cv2.countNonZero(mask)`（与阶段 3 §4.3 对齐） | 可 `None` |
| `severity` | 恒为 `None`（阶段 6 前预留） | 可 `None` |

后处理：模型内置 NMS（`iou_threshold`）+ 置信度过滤（`conf_threshold`）先执行，再将幸存框转换为 `Detection`；转换发生在 `detect` 内，统一输出 `List[Detection]`。

### 4.3 类别不平衡处理

- 训练数据中缺陷样本少：数据增强（旋转/缩放/亮度/合成缺陷）、类别加权、过采样/欠采样。

### 4.4 参数配置表（配置驱动）

所有以下字段必须通过 `load(cfg)` 传入，**禁止写死在代码里**。`cfg` 结构约定（由 `build_detector` 下发，与阶段 3 §4.4 同风格）：

```yaml
detector:
  type: yolo
  params:            # 落入 load(cfg)；字段说明与默认值见下表
    model_path: models/yolov8n.pt
    conf_threshold: 0.25
    iou_threshold: 0.45
    device: cpu      # 或 cuda / cuda:0 等
```

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `model_path` | str | 无（必填） | 权重文件路径；须为存在且可读的文件（`.pt`/`.onnx` 等）。缺失/不存在/不可读 → `DetectorLoadError` |
| `conf_threshold` | float | 0.25 | 置信度下限，`< conf_threshold` 的框丢弃；取值 `[0, 1]` |
| `iou_threshold` | float | 0.45 | NMS 的 IoU 阈值；取值 `[0, 1]` |
| `device` | str | `"cpu"` | 推理设备：`"cpu"` / `"cuda"` / `"cuda:N"`（`N ≥ 0`） |
| `imgsz`（可选） | int | 640 | 推理输入尺寸（边长，px）；须为正整数 |

校验规则（违反即抛 `DetectorLoadError`，见 §5.1）：

- `model_path` 非空 `str` 且文件存在、可读；否则 `DetectorLoadError`。
- `conf_threshold`、`iou_threshold` 为数值且 `∈ [0, 1]`。
- `device` 若含 `cuda`（`cuda` / `cuda:N`），则 `load` 时校验 `torch.cuda.is_available()` 与设备号 `N` 是否越界；不可用 → `DetectorLoadError`，**不静默回退到 `cpu`**（避免行为不可预期，§8 特殊风险点名）。
- `imgsz` 为正整数；`params` 出现未知字段 → `DetectorLoadError`。

## 5. 关键接口落位

`YoloDetector(Detector)` 实现阶段 0 §5.4.2 五个契约成员，签名与阶段 0 逐参数一致：

- `name`（`@property`）→ `str`，返回模型变体标识（如 `"yolov8n"`），不可 `None`。
- `version`（`@property`）→ `str`，记录权重版本（语义化版本，默认 `"0.1.0"`），供报告溯源，不可 `None`。
- `load(cfg: dict) -> None`：`cfg` 为 `config["detector"]`（含 `params`，见 §4.4）；读入参数、加载模型权重到指定 `device`，不启动输入源。返回 `None`。
- `detect(frame: Frame) -> List[Detection]`：对每帧执行 §4.2 流程；无缺陷返回空列表 `[]`，**不返回 `None`**。
- `release() -> None`：释放模型与显存（见 §5.2）；幂等。

`detect` 输出的每个 `Detection` 字段来源与可空性见 §4.2 转换表，严格符合阶段 0 §5.4.1；`detect` 返回 `List[Detection]`，字段名/类型/单位与 `OpencvDetector.detect` 完全一致（阶段 3 §8 验收 #5 的对照基准）。

### 5.1 异常语义

`detect(frame)`：

| 情形 | 预期行为 |
| --- | --- |
| `frame` 为 `None` | 抛 `ValueError` |
| `frame.image` 为 `None` 或非 `np.ndarray` | 抛 `ValueError` |
| `frame.image` 尺寸为 0（`height == 0` 或 `width == 0`） | 抛 `ValueError` |
| `frame.image.shape[:2]` 与 `frame.height` / `frame.width` 不一致 | 抛 `ValueError` |
| 有效帧但未检出缺陷 | 返回空列表 `[]`（不返回 `None`、不抛异常） |
| 推理失败（含 CUDA OOM、`RuntimeError`、`torch.cuda.OutOfMemoryError`） | 捕获并转译为 `DetectorInferenceError`（由调用方捕获），原始异常挂 `from exc` 保留堆栈 |

> 空帧、尺寸为 0 属「输入非法」抛 `ValueError`；「无缺陷」与「出错」不共用 `None` 返回，与阶段 0 §5.4.2 契约一致。单条结果 `class_id` 越界不视为推理失败，仅丢弃该条并记日志，不影响整帧返回。

`load(cfg)`：

| 情形 | 预期行为 |
| --- | --- |
| `cfg` 非 `dict` 或缺少 `params` | 抛 `DetectorLoadError`（由调用方捕获） |
| `params` 含未知字段 | 抛 `DetectorLoadError` |
| `model_path` 未提供 / 文件不存在 / 不可读 | 抛 `DetectorLoadError` |
| 模型格式错误 / 加载失败（反序列化失败） | 抛 `DetectorLoadError` |
| `device` 指定 `cuda` 但不可用 | 抛 `DetectorLoadError` |
| 参数值非法（阈值越界、`imgsz ≤ 0` 等） | 抛 `DetectorLoadError` |
| 参数合法 | 返回 `None` |

`release()`：幂等；重复调用不抛异常、不重复释放（见 §5.2）。

### 5.2 并发与资源生命周期

- **模型资源生命周期**：`load` 创建模型对象并把权重加载到 `device`（`cuda` 时分配显存）→ `detect` 使用（评价模式 `model.eval()` + `torch.no_grad()`）→ `release` 释放。
- **资源所有者**：`YoloDetector` 实例自身；「谁 `load` 谁 `release`」，不存在释放责任不清。
- **释放动作**：`release()` 中先 `del` 模型引用并 `gc.collect()`，若 `device` 含 `cuda` 则调用 `torch.cuda.empty_cache()` 释放未用显存；再次进入 `release()` 直接返回，不重复执行。
- **异常路径释放**：`load` 中途失败（如权重加载到显存时 OOM）需清理半初始化状态后再向上抛 `DetectorLoadError`；`detect` 内推理异常**不**释放模型（模型仍有效，可重试），由调用方决定是否 `release`。释放动作统一置于调用侧的 `try/finally` 契约约定（与阶段 0 `release` 幂等要求一致）。
- **线程模型**：**单线程**。`detect` 每次调用只读 `load` 后缓存的配置与模型权重，不修改可变共享状态；不支持并发调用，若需并发由调用方自建独立实例（各自持有独立模型/显存）。

## 6. 交付物

- YOLO 检测器（可插拔）。
- 训练/推理脚本 + 标注格式说明。
- 评测报告（与 OpenCV 对照）。
- 参数配置样例（`configs/yolo_demo.yaml`，含 §4.4 全部字段）。
- 单元测试（`tests/test_yolo_detector.py`）：字段契约、异常语义、生命周期（含 `release` 幂等）、配置校验、映射表一致性。

## 7. 阶段要求与验收标准

| # | 验收标准 | 通过判定（可执行命令 / 可观测结果） |
| --- | --- | --- |
| 1 | 同一批样本上 YOLO 与 OpenCV 输出同构 `Detection`，可在配置/界面切换 | 运行 `python3 main.py --config configs/demo.yaml`（`type: opencv`）与 `--config configs/yolo_demo.yaml`（`type: yolo`），两者返回的 `List[Detection]` 字段名/类型/单位一致；`detector.type` 不同即可切换 |
| 2 | YOLO 对复杂/多样性缺陷的召回优于纯传统方法 | 运行评测脚本（§6 交付物②），同一测试集上 `recall@YOLO > recall@OpenCV`，并输出 mAP/精确率对照表 |
| 3 | `release()` 正确释放显存，推理 FPS 满足既定额度 | 运行一轮 `detect` 后调用 `release()`，`nvidia-smi` 显存回到 `load` 前水位（或 `torch.cuda.memory_allocated()` 归零）；`cpu` 设备下该条以「无可观测显存」判定通过 |
| 4 | 训练、推理、评测流程可复现 | 保存超参 + 随机种子，重跑评测得到一致 mAP（`±0.001` 容差） |
| 5 | `load`/`detect`/`release` 契约与异常语义正确 | 运行 `python3 -m pytest -q tests/test_yolo_detector.py`：模型文件缺失→`DetectorLoadError`、空帧→`ValueError`、模拟推理失败→`DetectorInferenceError`、`release` 重复调用不抛异常 |

## 8. 风险与注意事项

- 缺陷样本稀缺是本路线最大瓶颈，先做数据增强与预训练权重迁移。
- GPU 显存受限时选小模型（n/s 量级）或量化。
- 严格保证类别 id 与 OpenCV 一致，否则报告/统计错乱。

特殊风险（固定格式，仅点名不展开）：

- 【YoloDetector】在【框级检测无像素掩码】存在【`area_px` 只能以 bbox 面积近似、与阶段 0「缺陷实际像素面积」口径存在偏差的风险】。
- 【YoloDetector】在【类别 id 依赖训练标注顺序】存在【训练 `class_id` 与阶段 3 §4.1 映射表不一致导致报告/统计错乱的风险】。
- 【YoloDetector】在【`device` 参数指定 `cuda`】存在【运行时设备不可用而启动失败、或静默回退导致行为不可预期的风险】。

## 9. 与其他阶段衔接

- 阶段 3 的 `OpencvDetector` 与本检测器并列，通过同一 `Detector` 接口被主流程调用；本阶段训练标注的类别 id 复用阶段 3 §4.1 映射表（交叉核对一致）。
- 阶段 5 可视化与判定同时消费本检测器与 OpenCV 的输出 `List[Detection]`，依赖 `bbox`（像素）/ `confidence` / `class_id` / `class_name` 字段。
- 阶段 6 在 YOLO 迁移+增强基础上做小样本/无监督扩展，YOLO 作为深度学习路线代表参评。
- 阶段 7 报告复用 `class_name` / `area_px` / `area_ratio` / `confidence` 字段，口径应与本文 §4.2 一致（其中 `area_px` 为 bbox 面积近似，需在报告中如实标注）。