# 双算法对比报告（OpenCV vs YOLO best.pt）

- 生成时间：2026-10-21 16:17:26
- 数据集：`/home/yjh/demo_OpenCv_picture/datasets/neu_det/images/val`（取样方式：stratified，共 30 张）
- 判定配置：`max_count=0`, `max_area_ratio=0.01`, `min_confidence=0.25`, `min_area_px=10`（两算法共用）
- 说明：仅统计检测数据，不绘制标注图；耗时为单帧 `detect()` 调用，已排除模型加载与首次预热。

## 汇总

| 指标 | OpenCV | YOLO(best.pt) |
| --- | --- | --- |
| 样本数 | 30 | 30 |
| PASS | 18 | 4 |
| FAIL | 12 | 26 |
| 平均检测数/帧 | 1.00 | 2.17 |
| 平均单帧耗时 | 3.7 ms | 29.2 ms |
| 中位单帧耗时 | 3.7 ms | 23.7 ms |
| 最大单帧耗时 | 4.5 ms | 41.0 ms |

## 判定一致性

| 指标 | 数值 |
| --- | --- |
| 一致样本数 | 14/30 |
| 一致率 | 46.7% |
| 同为 PASS | 3 |
| 同为 FAIL | 11 |
| OpenCV PASS → YOLO FAIL | 15 |
| OpenCV FAIL → YOLO PASS | 1 |

## 逐图明细

| # | 图片 | OpenCV 检测数 | OpenCV 耗时(ms) | OpenCV 判定 | YOLO 检测数 | YOLO 耗时(ms) | YOLO 判定 | 一致 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | crazing_10.jpg | 1 | 4.4 | FAIL | 0 | 39.5 | PASS | 否 |
| 2 | inclusion_10.jpg | 0 | 3.7 | PASS | 4 | 41.0 | FAIL | 否 |
| 3 | patches_1.jpg | 2 | 4.5 | FAIL | 3 | 40.2 | FAIL | 是 |
| 4 | pitted_surface_102.jpg | 0 | 3.7 | PASS | 3 | 39.9 | FAIL | 否 |
| 5 | rolled-in_scale_105.jpg | 3 | 4.0 | FAIL | 1 | 40.1 | FAIL | 是 |
| 6 | scratches_1.jpg | 0 | 3.0 | PASS | 1 | 40.1 | FAIL | 否 |
| 7 | crazing_103.jpg | 0 | 4.5 | PASS | 0 | 40.2 | PASS | 是 |
| 8 | inclusion_101.jpg | 0 | 3.4 | PASS | 2 | 40.9 | FAIL | 否 |
| 9 | patches_106.jpg | 4 | 3.8 | FAIL | 1 | 40.2 | FAIL | 是 |
| 10 | pitted_surface_109.jpg | 0 | 3.5 | PASS | 1 | 38.6 | FAIL | 否 |
| 11 | rolled-in_scale_11.jpg | 0 | 4.1 | PASS | 1 | 38.6 | FAIL | 否 |
| 12 | scratches_100.jpg | 1 | 3.2 | FAIL | 1 | 38.6 | FAIL | 是 |
| 13 | crazing_104.jpg | 1 | 4.1 | FAIL | 1 | 28.3 | FAIL | 是 |
| 14 | inclusion_11.jpg | 0 | 3.0 | PASS | 3 | 20.3 | FAIL | 否 |
| 15 | patches_107.jpg | 1 | 4.1 | FAIL | 2 | 20.0 | FAIL | 是 |
| 16 | pitted_surface_11.jpg | 1 | 3.5 | FAIL | 2 | 21.0 | FAIL | 是 |
| 17 | rolled-in_scale_114.jpg | 0 | 3.8 | PASS | 2 | 23.1 | FAIL | 否 |
| 18 | scratches_112.jpg | 0 | 3.0 | PASS | 5 | 20.5 | FAIL | 否 |
| 19 | crazing_116.jpg | 0 | 4.1 | PASS | 0 | 20.0 | PASS | 是 |
| 20 | inclusion_118.jpg | 5 | 3.3 | FAIL | 3 | 21.1 | FAIL | 是 |
| 21 | patches_111.jpg | 0 | 3.9 | PASS | 4 | 22.0 | FAIL | 否 |
| 22 | pitted_surface_111.jpg | 4 | 3.3 | FAIL | 2 | 20.8 | FAIL | 是 |
| 23 | rolled-in_scale_117.jpg | 0 | 4.0 | PASS | 1 | 20.4 | FAIL | 否 |
| 24 | scratches_113.jpg | 0 | 3.1 | PASS | 5 | 20.9 | FAIL | 否 |
| 25 | crazing_120.jpg | 0 | 4.4 | PASS | 0 | 20.4 | PASS | 是 |
| 26 | inclusion_126.jpg | 0 | 3.6 | PASS | 5 | 24.2 | FAIL | 否 |
| 27 | patches_112.jpg | 0 | 3.9 | PASS | 5 | 30.9 | FAIL | 否 |
| 28 | pitted_surface_118.jpg | 4 | 3.5 | FAIL | 1 | 21.5 | FAIL | 是 |
| 29 | rolled-in_scale_118.jpg | 3 | 3.7 | FAIL | 3 | 21.2 | FAIL | 是 |
| 30 | scratches_127.jpg | 0 | 3.0 | PASS | 3 | 21.4 | FAIL | 否 |
