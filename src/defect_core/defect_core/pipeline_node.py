"""检测流水线 ROS2 节点（实现文档 阶段1 骨架）。

订阅图像话题 → 执行流水线 → 发布 defect_interfaces/DefectArray，
并提供 /set_detector 服务实现运行时算法切换。
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from defect_interfaces.msg import Defect, DefectArray
from defect_interfaces.srv import SetDetector

from . import detectors  # noqa: F401  导入即完成算法注册
from .image_utils import imgmsg_to_numpy
from .pipeline import Pipeline
from .registry import available_detectors, create_detector
from .types import Detection, Frame


class PipelineNode(Node):
    """订阅图像、执行检测、发布缺陷结果。"""

    def __init__(self) -> None:
        super().__init__('defect_pipeline_node')

        self.declare_parameter('detector', 'opencv')
        self.declare_parameter('image_topic', 'image_raw')
        self.declare_parameter('result_topic', 'defects')

        detector_name = self.get_parameter('detector').value
        self._detector = create_detector(detector_name)
        self._detector.load({})
        self._pipeline = Pipeline(self._detector)

        image_topic = self.get_parameter('image_topic').value
        result_topic = self.get_parameter('result_topic').value

        self._sub = self.create_subscription(Image, image_topic, self._on_image, 10)
        self._pub = self.create_publisher(DefectArray, result_topic, 10)
        self._srv = self.create_service(SetDetector, 'set_detector', self._on_set_detector)

        self.get_logger().info(
            f'流水线已启动：算法={detector_name}，可用算法={available_detectors()}')

    def _on_image(self, msg: Image) -> None:
        image = imgmsg_to_numpy(msg)
        frame = Frame(
            image=image,
            width=msg.width,
            height=msg.height,
            source_id=msg.header.frame_id,
            timestamp=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
        )

        result = self._pipeline.run(frame)

        out = DefectArray()
        out.header = msg.header
        out.source_id = result.source_id
        out.detector_name = result.detector_name
        out.detector_version = result.detector_version
        out.is_pass = result.is_pass
        out.total_defects = result.total_defects
        out.defects = [self._to_msg(d) for d in result.defects]
        self._pub.publish(out)

    @staticmethod
    def _to_msg(d: Detection) -> Defect:
        msg = Defect()
        msg.class_id = d.class_id
        msg.class_name = d.class_name
        msg.x, msg.y, msg.width, msg.height = (int(v) for v in d.bbox)
        msg.area_px = float(d.area_px)
        msg.area_ratio = float(d.area_ratio)
        msg.confidence = float(d.confidence)
        msg.severity = int(d.severity)
        return msg

    def _on_set_detector(self, request: SetDetector.Request, response: SetDetector.Response):
        """运行时切换算法：失败时回滚到原算法，不影响当前任务。"""
        try:
            detector = create_detector(request.detector_name)
            detector.load({})
        except (KeyError, RuntimeError) as exc:
            response.success = False
            response.message = str(exc)
            return response

        old = self._detector
        self._detector = detector
        self._pipeline.set_detector(detector)
        old.release()

        response.success = True
        response.message = f'已切换至 {request.detector_name}'
        self.get_logger().info(response.message)
        return response

    def destroy_node(self) -> bool:
        self._detector.release()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PipelineNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
