# 单个缺陷检测结果（对应实现文档阶段0 的 Detection 数据结构）

# 缺陷类别常量
uint8 CLASS_UNKNOWN = 0
uint8 CLASS_SCRATCH = 1   # 划痕
uint8 CLASS_DENT = 2      # 凹坑
uint8 CLASS_STAIN = 3     # 污渍
uint8 CLASS_NOTCH = 4     # 缺口

uint8 class_id            # 缺陷类别 id
string class_name         # 缺陷类别名称

# 边界框，图像像素坐标系（左上角为原点）
uint32 x
uint32 y
uint32 width
uint32 height

float32 area_px           # 像素面积
float32 area_ratio        # 相对图像面积占比，0.0~1.0
float32 confidence        # 置信度，0.0~1.0
uint8 severity            # 严重程度等级（阶段6 启用，阶段0 预留）
