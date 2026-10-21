# 双算法对比报告（OpenCV vs YOLO best.pt）

- 生成时间：2026-10-21 20:53:35
- 数据集：`/home/yjh/demo_OpenCv_picture/datasets/neu_det/images/val`（取样方式：stratified，共 30 张）
- 判定配置：`max_count=0`, `max_area_ratio=0.01`, `min_confidence=0.25`, `min_area_px=10`（两算法共用）
- 说明：仅统计检测数据，不绘制标注图；耗时为单帧 `detect()` 调用，已排除模型加载与首次预热。

## 汇总

| 指标 | OpenCV | YOLO(best.pt) |
| --- | --- | --- |
| 样本数 | 30 | 30 |
| PASS | 18 | 2 |
| FAIL | 12 | 28 |
| 平均检测数/帧 | 1.00 | 2.70 |
| 平均单帧耗时 | 3.7 ms | 64.8 ms |
| 中位单帧耗时 | 3.6 ms | 48.6 ms |
| 最大单帧耗时 | 4.4 ms | 102.4 ms |

## 判定一致性

| 指标 | 数值 |
| --- | --- |
| 一致样本数 | 14/30 |
| 一致率 | 46.7% |
| 同为 PASS | 2 |
| 同为 FAIL | 12 |
| OpenCV PASS → YOLO FAIL | 16 |
| OpenCV FAIL → YOLO PASS | 0 |

## 逐图明细

| # | 图片 | OpenCV 检测数 | OpenCV 耗时(ms) | OpenCV 判定 | YOLO 检测数 | YOLO 耗时(ms) | YOLO 判定 | 一致 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | crazing_10.jpg | 1 | 4.4 | FAIL | 1 | 65.1 | FAIL | 是 |
| 2 | inclusion_10.jpg | 0 | 3.5 | PASS | 5 | 84.2 | FAIL | 否 |
| 3 | patches_1.jpg | 2 | 4.3 | FAIL | 3 | 102.0 | FAIL | 是 |
| 4 | pitted_surface_102.jpg | 0 | 3.6 | PASS | 2 | 102.4 | FAIL | 否 |
| 5 | rolled-in_scale_105.jpg | 3 | 3.9 | FAIL | 1 | 102.3 | FAIL | 是 |
| 6 | scratches_1.jpg | 0 | 2.9 | PASS | 1 | 102.1 | FAIL | 否 |
| 7 | crazing_103.jpg | 0 | 4.2 | PASS | 0 | 96.1 | PASS | 是 |
| 8 | inclusion_101.jpg | 0 | 3.3 | PASS | 2 | 95.4 | FAIL | 否 |
| 9 | patches_106.jpg | 4 | 3.8 | FAIL | 1 | 73.0 | FAIL | 是 |
| 10 | pitted_surface_109.jpg | 0 | 3.5 | PASS | 4 | 82.9 | FAIL | 否 |
| 11 | rolled-in_scale_11.jpg | 0 | 4.0 | PASS | 1 | 86.3 | FAIL | 否 |
| 12 | scratches_100.jpg | 1 | 3.2 | FAIL | 2 | 81.5 | FAIL | 是 |
| 13 | crazing_104.jpg | 1 | 4.1 | FAIL | 3 | 49.8 | FAIL | 是 |
| 14 | inclusion_11.jpg | 0 | 3.0 | PASS | 4 | 47.5 | FAIL | 否 |
| 15 | patches_107.jpg | 1 | 4.1 | FAIL | 2 | 48.1 | FAIL | 是 |
| 16 | pitted_surface_11.jpg | 1 | 3.4 | FAIL | 4 | 48.2 | FAIL | 是 |
| 17 | rolled-in_scale_114.jpg | 0 | 3.7 | PASS | 5 | 47.6 | FAIL | 否 |
| 18 | scratches_112.jpg | 0 | 3.0 | PASS | 5 | 47.5 | FAIL | 否 |
| 19 | crazing_116.jpg | 0 | 4.1 | PASS | 1 | 47.3 | FAIL | 否 |
| 20 | inclusion_118.jpg | 5 | 3.5 | FAIL | 3 | 47.7 | FAIL | 是 |
| 21 | patches_111.jpg | 0 | 3.9 | PASS | 3 | 48.0 | FAIL | 否 |
| 22 | pitted_surface_111.jpg | 4 | 3.3 | FAIL | 4 | 47.7 | FAIL | 是 |
| 23 | rolled-in_scale_117.jpg | 0 | 4.0 | PASS | 1 | 47.7 | FAIL | 否 |
| 24 | scratches_113.jpg | 0 | 3.0 | PASS | 4 | 48.0 | FAIL | 否 |
| 25 | crazing_120.jpg | 0 | 4.4 | PASS | 0 | 47.4 | PASS | 是 |
| 26 | inclusion_126.jpg | 0 | 3.5 | PASS | 4 | 47.8 | FAIL | 否 |
| 27 | patches_112.jpg | 0 | 3.8 | PASS | 5 | 47.6 | FAIL | 否 |
| 28 | pitted_surface_118.jpg | 4 | 3.5 | FAIL | 4 | 55.1 | FAIL | 是 |
| 29 | rolled-in_scale_118.jpg | 3 | 3.7 | FAIL | 3 | 48.5 | FAIL | 是 |
| 30 | scratches_127.jpg | 0 | 3.0 | PASS | 3 | 48.6 | FAIL | 否 |
