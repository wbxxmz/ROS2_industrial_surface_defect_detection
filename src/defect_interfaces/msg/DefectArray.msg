# 单帧/单图的检测结果汇总（对应实现文档阶段5 的判定结果）

std_msgs/Header header

string source_id          # 输入源标识（文件路径 / 摄像头名 / 帧序号）
string detector_name      # 算法名称，用于溯源
string detector_version   # 算法版本 / 权重版本，用于溯源

bool is_pass              # 单件产品判定：是否合格
uint32 total_defects      # 检出缺陷总数

Defect[] defects
