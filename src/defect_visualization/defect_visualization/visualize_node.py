"""可视化节点：订阅图像与缺陷结果，叠加展示（实现文档 阶段5）。"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from defect_core import (Detection, InspectionResult, imgmsg_to_numpy,
                         numpy_to_imgmsg)
from defect_interfaces.msg import DefectArray

from .judge import Judge
from .overlay import draw_detections, draw_stats


class VisualizeNode(Node):
    """订阅 image_raw 与 defects，绘制标注后图像并发布到 annotated 话题。"""

    def __init__(self) -> None:
        super().__init__('defect_visualize_node')

        self.declare_parameter('image_topic', 'image_raw')
        self.declare_parameter('result_topic', 'defects')
        self.declare_parameter('out_topic', 'annotated')

        self._last_result = None
        self._judge = Judge()

        self.create_subscription(
            Image, self.get_parameter('image_topic').value, self._on_image, 10)
        self.create_subscription(
            DefectArray, self.get_parameter('result_topic').value,
            self._on_result, 10)
        self._pub = self.create_publisher(
            Image, self.get_parameter('out_topic').value, 10)

        self.get_logger().info('可视化节点已启动')

    def _on_result(self, msg: DefectArray) -> None:
        """缓存最近一次检测结果，与图像帧联动叠绘。"""
        self._last_result = InspectionResult(
            source_id=msg.source_id,
            detector_name=msg.detector_name,
            detector_version=msg.detector_version,
            is_pass=msg.is_pass,
            defects=[Detection(
                class_id=d.class_id,
                class_name=d.class_name,
                bbox=(d.x, d.y, d.width, d.height),
                confidence=d.confidence,
                area_px=int(d.area_px),
                area_ratio=d.area_ratio,
                severity=d.severity,
            ) for d in msg.defects],
        )

    def _on_image(self, msg: Image) -> None:
        image = imgmsg_to_numpy(msg)

        annotated = image
        if self._last_result is not None:
            annotated = draw_detections(image, self._last_result.defects)
            annotated = draw_stats(annotated, self._last_result)

        out = numpy_to_imgmsg(
            annotated,
            encoding=msg.encoding,
            frame_id=msg.header.frame_id,
            stamp=msg.header.stamp,
        )
        self._pub.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VisualizeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
