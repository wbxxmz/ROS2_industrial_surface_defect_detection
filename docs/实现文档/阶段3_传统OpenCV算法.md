# 阶段 3：传统 OpenCV 算法实现 · 实现技术文档

## 1. 阶段目标

不依赖任何标注与训练，用传统图像处理实现四类缺陷（划痕/凹坑/污渍/缺口）的检测，输出与阶段 0 一致的 `Detection` 结构。这是 MVP 中"最快见效"的算法路线。

## 2. 前置条件与依赖

- 依赖阶段 0 的 `Detector` 接口、阶段 2 的帧输入。
- 依赖 OpenCV、NumPy。
- 有一批可用于调参的样本图片（含四类缺陷与正常样本）。

## 3. 工作分解（WBS）

1. 预处理：灰度化、去噪（高斯/中值）、光照校正（直方图均衡 / CLAHE）。
2. 通用特征提取：滤波、阈值（全局/自适应/Otsu）、边缘（Canny/Sobel）、形态学（开闭运算、顶帽/黑帽）。
3. 逐缺陷类型设计检测策略（见第 5 节）。
4. 后处理：连通域分析、面积/几何约束过滤、分类与定位。
5. 封装为 `OpencvDetector`，实现 `detect(frame) -> List[Detection]`。
6. 参数可配置化（阈值、形态学核、面积上下限等，供调参）。
7. 定义四类缺陷 `class_id` / `class_name` 映射表并写死（见 §4.1）。
8. 编写单元测试：字段契约、异常语义、边界过滤、置信度取值范围。

## 4. 关键技术方案

- 流水线：`预处理 → 特征/候选区域提取 → 过滤 → 分类 → Detection 组装`。
- 制定统一后处理规则：对小连通域去噪、按面积与长宽比/圆度等几何特征筛选，避免碎点误报。
- 面积计算：通过轮廓 / 掩码统计像素，填 `area_px` 与 `area_ratio`。
- 置信度近似：用规则打分（如特征响应强度、与阈值类的偏差程度）映射到 0–1，标注"规则置信度"（见 §4.2）。

### 4.1 类别映射表（写死，阶段 4 复用）

本表是四类缺陷 `class_id` 与 `class_name` 的**唯一权威映射**。阶段 4（YOLO）、阶段 5（可视化）、阶段 7（报告）必须复用本表，不得各自定义。

| class_id | class_name | 中文标签 | 说明 |
| --- | --- | --- | --- |
| 0 | scratch | 划痕 | 细长线状缺陷 |
| 1 | dent | 凹坑 | 点状/局部凹陷，多为暗斑 |
| 2 | stain | 污渍 | 颜色/纹理异常区域 |
| 3 | notch | 缺口 | 边缘破损 |

约束：

- `class_id` 为 `int`，`class_name` 为 `str`，二者一一对应，均不可为 `None`。
- 阶段 4 训练标注的类别 id 必须与上表一致（0=scratch、1=dent、2=stain、3=notch），不得重排或另起映射。

### 4.2 置信度规则（规则分数，不得为空）

`confidence` 为 `float`，取值范围 `[0, 1]`，恒不为 `None`。传统算法无模型概率，用**规则响应分数**近似，统一公式：

```
confidence = clip(score, 0.0, 1.0)
```

其中 `score` 为该候选在其类别检出路线的「响应强度」，四类口径如下：

| class_id | class_name | score 口径 | score 取值范围 |
| --- | --- | --- | --- |
| 0 | scratch | 方向性形态学响应（顶帽与黑帽取大）均值 / 255 | [0, 1] |
| 1 | dent | |LoG 响应| 均值 / 255 | [0, 1] |
| 2 | stain | 前景与背景的颜色距离归一化值 `d / d_max`（HSV/Lab 色差） | [0, 1] |
| 3 | notch | 最大凸包缺陷深度 / 凸包直径 | [0, 1] |

约束：

- 候选若无法计算出有效 `score`（如除零、掩码为空），视为无效候选，**不输出**该 `Detection`，绝不能以 `None` 或空值填充 `confidence`。
- `confidence < min_confidence`（默认 0.05）的候选被过滤，不输出。
- 该分数为"规则置信度"，仅表示**同一类型内部**的相对可信程度；不同类型间不保证可横向比较（§9 特殊风险已点名）。
- 划痕的形态学响应同时取顶帽与黑帽（明/暗两种极性）的较大者：暗色裂纹在顶帽下响应被低估，双极性取大可避免漏检（实现见 `_line_response`）。

### 4.3 面积与 bbox 口径

- `area_px`：`int`，取 `len(cv2.findContours...)` 所得轮廓的 `int(round(cv2.contourArea(cnt)))`——**实际像素面积**，不等于 bbox 矩形面积（细长划痕尤甚）。
- `area_ratio`：`float`，`area_px / (frame.width * frame.height)`，恒在 `[0, 1]`。因空帧/尺寸 0 已在 `detect` 入口抛 `ValueError`（见 §6），此处 `width*height > 0` 恒成立，无除零。
- `bbox`：`tuple[int, int, int, int]`，取 `cv2.boundingRect(cnt)` 的 `(x, y, w, h)`——像素、int、原点左上、x 向右、y 向下，`(x, y)` 为左上角，非归一化。
- `mask`：可选；本阶段定为 `np.ndarray`，`dtype=uint8`，尺寸与 `frame.image` 相同（H×W），取值 `0/255`，由 `cv2.drawContours` 填充缺陷轮廓得到。单通道二值掩码；不输出掩码时置 `None`。

> 本表同时补全阶段 0 §5.4.5 遗留的「mask 数据类型与格式 → TBD-阶段 3」：锁定为 `uint8` 单通道 `0/255` 掩码，尺寸等于帧图像尺寸。

### 4.4 参数配置表（配置驱动）

所有阈值参数必须通过 `load(cfg)` 传入，**禁止写死在代码里**。`cfg` 结构约定（由阶段 3 扩展后的 `build_detector` 下发，与阶段 2 输入源配置同风格）：

```yaml
detector:
  type: opencv          # 阶段 3 在 build_detector 中新增识别；此时 type=opencv
  params:               # 落入 load(cfg)；缺省字段沿用下方默认值
    min_area: 10
    # ... 其余字段
```

`load(cfg)` 内部用 `OpencvDetectorConfig.from_dict(cfg["params"])` 校验并转强类型（风格同阶段 2 `SourceConfig.from_dict`）；字段缺省用下表默认值，出现未知字段或非法值按 §6 抛 `DetectorLoadError`。

| 字段 | 类型 | 默认值 | 作用域 | 说明 |
| --- | --- | --- | --- | --- |
| `min_area` | int | 10 | 通用 | 最小轮廓面积（px），面积 `< min_area` 的轮廓视为噪点直接丢弃 |
| `min_confidence` | float | 0.05 | 通用 | 置信度下限，低于此值的候选不输出 |
| `denoise` | str | `"gaussian"` | 通用 | 去噪方式：`gaussian` / `median` |
| `denoise_kernel` | int | 5 | 通用 | 去噪核大小（奇数），如 `5` |
| `clahe_clip` | float | 2.0 | 通用 | CLAHE 对比度限幅（光照校正） |
| `clahe_grid` | int | 8 | 通用 | CLAHE 分块大小 |
| `scratch_canny_low` | int | 50 | 划痕 | Canny 低阈值 |
| `scratch_canny_high` | int | 150 | 划痕 | Canny 高阈值（须 ≥ `scratch_canny_low`） |
| `scratch_min_aspect` | float | 3.0 | 划痕 | 最小长宽比，剔除块状误报 |
| `scratch_max_width` | int | 8 | 划痕 | 最大线宽（px） |
| `dent_log_sigma` | float | 3.0 | 凹坑 | LoG 标准差 |
| `dent_roundness_min` | float | 0.6 | 凹坑 | 圆形度下限（4πA/P²），剔除细长杂讯 |
| `dent_radius_min` | int | 2 | 凹坑 | 最小半径（px） |
| `dent_radius_max` | int | 20 | 凹坑 | 最大半径（px） |
| `stain_sat_min` | int | 60 | 污渍 | HSV 饱和度下限 |
| `stain_color_dist_min` | float | 30.0 | 污渍 | 前景与背景的最小颜色距离 |
| `stain_area_ratio_min` | float | 0.001 | 污渍 | 最小面积占比（相对整图） |
| `notch_convexity_min` | float | 0.02 | 缺口 | 凸包缺陷深度占凸包直径的最小比例 |

校验规则（违反即抛 `DetectorLoadError`，见 §6）：

- `min_area ≥ 1`；`0 ≤ min_confidence ≤ 1`。
- `scratch_canny_low ≤ scratch_canny_high`。
- `dent_radius_min ≤ dent_radius_max`。
- `denoise_kernel` 为奇数且 `≥ 1`。
- 其余数值字段须在各自表意范围内（如 `clahe_clip > 0`、`stain_area_ratio_min ∈ [0,1]`）。

### 4.5 边界与后处理过滤

- **去噪下限**：连通域分析后，面积 `< min_area`（默认 `10` px）的轮廓直接丢弃，避免输出过多碎片结果。
- **几何过滤**：按缺陷类型施加长宽比（划痕）、圆形度（凹坑）、面积占比（污渍）、凸包缺陷比例（缺口）约束，剔除与目标形态不符的杂讯。
- **整框面积上限（硬性过滤）**：`bbox` 矩形面积 `w × h` 超过整图面积（`frame.width × frame.height`）的 `15%` 时，该检测直接丢弃。该上限为硬编码常量（`MAX_BBOX_AREA_RATIO = 0.15`），不经 `load(cfg)` 传入。
- **非极大值抑制（可选，本阶段未实现）**：同一缺陷被多条链路重复检出时，保留置信度最高的一条；重复框 IoU 阈值统一取 `0.5`。当前各链路基于互不重叠的连通域，未出现重复框，故代码暂未启用。

## 5. 各缺陷类型检测思路（粗方案）

| 缺陷 | class_id | 思路 | 关键算子 | 置信度 score 口径 |
| --- | --- | --- | --- | --- |
| 划痕 | 0 | 细长线状，低/高对比 | 方向性滤波、Canny 边缘 + 直线/轮廓形态筛选、顶帽/黑帽变换（明暗双极性） | 方向性形态学响应（顶帽/黑帽取大）均值 / 255 |
| 凹坑 | 1 | 点状/局部凹陷，多为暗斑 | 局部阈值/分水岭、圆形度筛选、高斯拉普拉斯(LoG) | |LoG 响应| 均值 / 255 |
| 污渍 | 2 | 颜色/纹理异常区域 | 颜色空间转换(HSV/Lab)阈值、差分、blob 检测 | 前景-背景色差归一化 |
| 缺口 | 3 | 边缘破损 | 边缘轮廓完整性分析、与模板/标准轮廓比对、凸包缺陷 | 最大凸包缺陷深度 / 凸包直径 |

> 注：本阶段无需追求完美，先把四类各跑通一条可用的检出链路。

## 6. 关键接口落位

`OpencvDetector(Detector)` 实现阶段 0 §5.4.2 五个契约成员，签名与阶段 0 逐参数一致：

- `name`（`@property`）→ `str`，恒返回 `"opencv"`，不可 `None`。
- `version`（`@property`）→ `str`，语义化版本 `"0.1.0"`，不可 `None`。
- `load(cfg: dict) -> None`：`cfg` 为 `config["detector"]`（含 `params`，见 §4.4）；读入并缓存参数，不启动任何外部资源。返回 `None`。
- `detect(frame: Frame) -> List[Detection]`：对每帧执行 §4 流程；无缺陷返回空列表 `[]`，**不返回 `None`**。
- `release() -> None`：释放资源；本检测器无持久外部资源（无模型权柄、无文件/摄像头句柄），实现为幂等空操作。

`detect` 输出的每个 `Detection` 字段严格符合阶段 0 §5.4.1：

| 字段 | 值来源 | 是否可 None |
| --- | --- | --- |
| class_id / class_name | §4.1 映射表 | 不可 None |
| bbox | `cv2.boundingRect`：`(x,y,w,h)` 像素 int、左上原点 | 不可 None |
| mask | `cv2.drawContours` 生成的 uint8 0/255 掩码 | 可 None（不输出掩码时） |
| area_px | `int(round(cv2.contourArea))` 实际像素面积 | 不可 None |
| area_ratio | `area_px / (width*height)` | 不可 None |
| confidence | §4.2 规则分数，clip 到 [0,1] | 不可 None |
| severity | 恒为 `None`（阶段 6 前预留） | 可 None（可选） |

### 6.1 异常语义

`detect(frame)`：

| 情形 | 预期行为 |
| --- | --- |
| `frame` 为 `None` | 抛 `ValueError` |
| `frame.image` 为 `None` 或非 `np.ndarray` | 抛 `ValueError` |
| `frame.image` 尺寸为 0（`height == 0` 或 `width == 0`） | 抛 `ValueError` |
| `frame.image.shape[:2]` 与 `frame.height` / `frame.width` 不一致 | 抛 `ValueError` |
| 有效帧但未检出缺陷 | 返回空列表 `[]`（不返回 `None`、不抛异常） |
| OpenCV 调用内部异常 | 抛 `DetectorInferenceError`（由调用方捕获） |

> 空帧、尺寸为 0 属「输入非法」，一律抛 `ValueError`；「无缺陷」与「出错」不共用 `None` 返回。二者语义与阶段 0 §5.4.2 契约完全一致。

`load(cfg)`：

| 情形 | 预期行为 |
| --- | --- |
| `cfg` 非 `dict` 或缺少 `params` | 抛 `DetectorLoadError`（由调用方捕获） |
| `cfg["params"]` 含未知字段 | 抛 `DetectorLoadError` |
| 参数值非法（如 `scratch_canny_low > scratch_canny_high`、`min_area ≤ 0`、`min_confidence ∉ [0,1]`） | 抛 `DetectorLoadError` |
| 参数合法 | 返回 `None` |

`release()`：幂等；重复调用不抛异常、不重复释放。

### 6.2 资源生命周期与线程模型

- 本检测器**无持久外部资源**：不持有文件句柄、`VideoCapture`、摄像头或模型显存，故 `load`/`detect` 期间无需 open→close；`release()` 为幂等空操作，仅预留接口一致性。
- 线程模型：**单线程**。`detect` 每次调用只读已缓存的配置（`load` 后只读），不修改可变共享状态；不支持并发调用，若需并发由调用方自建独立实例。

## 7. 交付物

- OpenCV 检测器（实现统一接口，`detectors/opencv_detector.py`）。
- 参数配置文件 + 调参说明（`configs/demo.yaml`）。
- 单张/批量演示脚本（扩展 `main.py --config` 支持 `type: opencv`）。
- 单元测试（`tests/test_opencv_detector.py`）：字段契约、异常语义、边界过滤、置信度取值范围、映射表一致性。

## 8. 阶段要求与验收标准

| # | 验收标准 | 通过判定（可执行命令 / 可观测结果） |
| --- | --- | --- |
| 1 | 在给定样本上四类缺陷均有可接受检出 | 运行 `python3 main.py --config configs/demo.yaml`，输出 `Detection` 的 `class_id` 集合 ⊇ `{0,1,2,3}` |
| 2 | 正常样本无明显碎片误报 | 对正常样本运行检测，输出为空列表，或每条 `area_px ≥ min_area` 且无面积 `< 10` px 的碎片 |
| 3 | `Detection` 字段完整且口径正确 | 运行 `python3 -m pytest -q tests/test_opencv_detector.py`，断言：`confidence ∈ [0,1]` 且非 `None`、`area_px` 为 `int` 且 ≥ `min_area`、`bbox` 为 4 元 `int` |
| 4 | 参数可配置，无需改代码 | 仅改 `configs/demo.yaml` 中参数（如 `scratch_canny_low`）后重跑同一图，检出结果发生变化（可日志比对）；全程不改 `.py` |
| 5 | 与阶段 4 YOLO 结果结构同构 | `OpencvDetector.detect` 与 `YoloDetector.detect` 均返回 `List[Detection]`，字段名/类型/单位一致（阶段 4 对照验证） |

## 9. 风险与注意事项

- 传统算法对光照/背景敏感，需强调预处理与打光配合，否则误报难以控制。
- 均为规则阈值，泛化弱——本阶段目标是"可用"，不是"通用"，通用交给 YOLO 与创新阶段。
- 调参成本高，建议参数集中到配置文件并记录样本-参数对应关系。

特殊风险（固定格式，仅点名不展开）：

- 【OpencvDetector】在【四类缺陷独立的规则打分】存在【各类 `confidence` 分数口径不一、不可横向比较的风险】。
- 【OpencvDetector】在【CLAHE 光照校正】存在【背景纹理被过度增强从而误判为污渍的风险】。
- 【OpencvDetector】在【凸包对齐的缺口判定】存在【对不同姿态/缩放的目标鲁棒性不足的风险】。

## 10. 与其他阶段衔接

- 阶段 4 的 YOLO 检测器与本检测器并列，通过同一 `Detector` 接口被主流程调用；其标注类别 id 复用 §4.1 映射表。
- 阶段 5 在 OpenCV 结果上做可视化与判定，直接消费 `Detection` 的 bbox / 面积 / 置信度 / `class_id`。
- 阶段 6 的算法对比，OpenCV 作为其中一个参评算法。
- 阶段 7 报告复用 `class_name` / `area_px` / `area_ratio` / `confidence` 字段，口径与 §4.3 / §4.2 一致。