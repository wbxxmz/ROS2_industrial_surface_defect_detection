# 扩展 02：模型注意力热力图（EigenCAM 为主 + 密度热力图兜底）

> 版本：V1.1 ｜ 所属模块：可视化增强（扩展计划总表 V1.1）
> 交付形态：附加可视化产物，不进入检测 / 判定 / 报告主链路。
> 说明：原方案 Grad-CAM 因依赖 YOLO 检测头的梯度反向传播、风险高，已降级为「选做 / 后续」，本轮主方案改为 **EigenCAM**，并以**检测密度热力图**作确定性兜底。

## 1. 目标

- 为检测结果叠加**可视化热力图**，直观展示「缺陷区域 / 模型关注区域」，作为答辩 / 面试的可视化亮点。
- 输出与原图融合的红蓝渐变图片，落盘 `reports/visual/`。

## 2. 契约影响

- **不改变 `Detection` 字段**，热力图仅作为附加可视化产物生成（符合扩展原则 P1）。
- 复用现有 `visualization/` 渲染产物与原图，不新增对外公开接口；`heatmap.py` 以独立脚本 / 内部函数形式暴露。
- 依赖现状满足：`ultralytics` 暴露模型结构与 forward hook 能力；无 CUDA / YOLO 未加载时沿用 `DetectorLoadError` 语义。
- 热力图生成失败仅**告警、不中断主流程**，保证检测与报告不受影响。

## 3. 核心方案（三档，按执行顺序）

新增 `visualization/heatmap.py`，内部按下列顺序实现 / 降级：

1. **兜底：检测密度热力图（本轮先行，零模型依赖）**
   - 对每条 `Detection` 的 `bbox` 做**高斯核叠加**（或对 `mask` 累加），生成「缺陷空间分布」红蓝热力图。
   - 100% 确定性、无 YOLO 内部依赖、十几分钟可打通；不经过模型 forward，天然无 CUDA 依赖。

2. **主方案：EigenCAM（梯度无关，forward-only）**
   - 对 YOLO backbone 最后一层卷积注册 **forward hook** 取末层特征图，按**通道加权**（梯度无关）得到注意力热力图，不做 backward、不选 target。
   - 保留「模型注意力热力图」的故事线，但绕开 Grad-CAM 最脆弱的反向传播与多框多类 target 选择。

3. **选做 / 后续：Grad-CAM（梯度版，本轮不承诺）**
   - 需 backward 反传 + 选定反传目标，风险高，仅作面试可讲的「进阶对比」或后续独立立项。

融合流程（两组热力图共用）：热力图 normalize 到 `[0,1]` → 上采样到原图尺寸 → JET 调色板得红蓝渐变 → `cv2.addWeighted` 与原图按 `α≈0.5` 融合。

输出与入口：
- 密度热力图：`reports/visual/{stem}_density.jpg`；EigenCAM：`reports/visual/{stem}_eigencam.jpg`。
- `{stem}` 复用 `ImageSource.current_path` 的真实文件名（与样本命名口径一致，见 Project Memory）。
- 入口：独立脚本 `scripts/render_heatmap.py --image xxx.jpg`，一条命令验收，避免侵入 `main.py` 主流程。

## 4. 验收标准

- 指定一张图运行，`reports/visual/` 下**同时产出密度热力图与 EigenCAM 两张图**（`{stem}_density.jpg` + `{stem}_eigencam.jpg`）。
- 打开图片能看到**红蓝渐变热力图**，且热区与缺陷位置（原框图）大致重合。
- 密度热力图在无 CUDA、无 YOLO 模型时也可独立生成（零模型依赖）；EigenCAM 依赖 YOLO，无 CUDA 时不崩溃、给出明确告警 / 报错。
- 不破坏既有检测与报告流程。

## 5. 风险说明

- **EigenCAM 仍依赖 `ultralytics` 的层索引**：hook 的 backbone 末层卷积下标与 `ultralytics` 版本绑定，**升级 / 更换 ultralytics 版本后需重新确认 hook 层**，否则会拿到错层特征或直接报错。
- Grad-CAM（梯度版）的 backward 在 YOLO 检测头上存在多框 / 多类 target 选择与梯度不稳定问题，本轮不承诺、不阻塞主流程。