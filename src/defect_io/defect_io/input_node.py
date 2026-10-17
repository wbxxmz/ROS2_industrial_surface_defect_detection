"""输入节点：读取输入源并以 sensor_msgs/Image 发布（实现文档 阶段2 · 4.2）。"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from defect_core import numpy_to_imgmsg

from .source_factory import create_input_source


class InputNode(Node):
    """把图片 / 视频 / 摄像头统一转换为 ROS2 图像话题。"""

    def __init__(self) -> None:
        super().__init__('defect_input_node')

        self.declare_parameter('source_type', 'camera')
        self.declare_parameter('source_path', '')
        self.declare_parameter('frame_stride', 1)
        self.declare_parameter('out_topic', 'image_raw')
        self.declare_parameter('publish_rate', 10.0)

        source_type = self.get_parameter('source_type').value
        self._source = create_input_source(source_type)
        self._source.open({
            'path': self.get_parameter('source_path').value,
            'source': self.get_parameter('source_path').value,
            'frame_stride': self.get_parameter('frame_stride').value,
        })

        out_topic = self.get_parameter('out_topic').value
        self._pub = self.create_publisher(Image, out_topic, 10)

        # 图片批量模式用定时器逐张发布；视频/摄像头由定时器驱动逐帧读取
        rate = self.get_parameter('publish_rate').value
        self._timer = self.create_timer(1.0 / rate, self._on_timer)
        self.get_logger().info(f'输入节点已启动：{source_type} → {out_topic}')

    def _on_timer(self) -> None:
        frame = self._source.read()
        if frame is None:
            self.get_logger().info('输入源已结束，停止发布')
            self._timer.cancel()
            return

        msg = numpy_to_imgmsg(
            frame.image,
            encoding='bgr8',
            frame_id=frame.source_id,
            stamp=self.get_clock().now().to_msg(),
        )
        self._pub.publish(msg)

    def destroy_node(self) -> bool:
        self._source.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = InputNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
