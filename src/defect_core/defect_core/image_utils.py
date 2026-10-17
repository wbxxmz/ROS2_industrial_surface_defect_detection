"""图像与 ROS2 消息互转工具（不依赖 cv_bridge）。

说明：ROS2 Humble 自带 cv_bridge 按 NumPy 1.x 编译，与本机 NumPy 2.x 不兼容，
故本平台自行实现 numpy <-> sensor_msgs/Image 转换，避免版本冲突。
"""

import numpy as np
from sensor_msgs.msg import Image

# 支持的编码 → 通道数
_ENCODING_CHANNELS = {'bgr8': 3, 'rgb8': 3, 'mono8': 1, '8UC1': 1, '8UC3': 3}


def imgmsg_to_numpy(msg: Image) -> np.ndarray:
    """把 sensor_msgs/Image 转为 numpy 数组。

    按 msg.step 重组二维形状，兼容带行填充（padding）的消息。
    """
    encoding = msg.encoding.lower()
    if encoding not in _ENCODING_CHANNELS:
        raise ValueError(f'暂不支持的图像编码: {msg.encoding}')

    channels = _ENCODING_CHANNELS[encoding]
    dtype = np.uint8
    itemsize = np.dtype(dtype).itemsize
    row_bytes = msg.width * channels * itemsize

    if msg.step != row_bytes:
        # 行末有填充：按 step 切行后截掉填充
        buffer = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.step)
        image = buffer[:, :row_bytes]
    else:
        image = np.frombuffer(msg.data, dtype=dtype).reshape(
            msg.height, msg.width * channels)

    if channels == 1:
        return image.reshape(msg.height, msg.width)
    return image.reshape(msg.height, msg.width, channels)


def numpy_to_imgmsg(image: np.ndarray, encoding: str = 'bgr8',
                    frame_id: str = '', stamp=None) -> Image:
    """把 numpy 数组转为 sensor_msgs/Image。"""
    encoding = encoding.lower()
    if encoding not in _ENCODING_CHANNELS:
        raise ValueError(f'暂不支持的图像编码: {encoding}')

    channels = _ENCODING_CHANNELS[encoding]
    if channels == 1 and image.ndim == 2:
        height, width = image.shape
    elif image.ndim == 3 and image.shape[2] == channels:
        height, width = image.shape[:2]
    else:
        raise ValueError(f'图像形状 {image.shape} 与编码 {encoding} 不匹配')

    # 保证内存连续，否则 tobytes 结果可能错乱
    image = np.ascontiguousarray(image, dtype=np.uint8)

    msg = Image()
    msg.height = int(height)
    msg.width = int(width)
    msg.encoding = encoding
    msg.is_bigendian = False
    msg.step = int(width * channels)
    msg.data = image.tobytes()
    msg.header.frame_id = frame_id
    if stamp is not None:
        msg.header.stamp = stamp
    return msg
