"""报告节点：订阅检测结果，累积后按批次生成报告（实现文档 阶段7）。"""

import rclpy
from rclpy.node import Node

from defect_core.types import Detection, InspectionResult
from defect_interfaces.msg import DefectArray

from .report_builder import ReportBuilder


class ReportNode(Node):
    """累积 DefectArray，达到批次数或超时后导出质检报告。"""

    def __init__(self) -> None:
        super().__init__('defect_report_node')

        self.declare_parameter('result_topic', 'defects')
        self.declare_parameter('batch_size', 100)
        self.declare_parameter('output_dir', 'reports')

        self._builder = ReportBuilder(
            {'output_dir': self.get_parameter('output_dir').value})
        self._results = []

        self._sub = self.create_subscription(
            DefectArray, self.get_parameter('result_topic').value, self._on_result, 10)
        self.get_logger().info('报告节点已启动')

    def _on_result(self, msg: DefectArray) -> None:
        self._results.append(InspectionResult(
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
        ))

        if len(self._results) >= int(self.get_parameter('batch_size').value):
            self.flush()

    def flush(self) -> None:
        """批量检测结束后汇总导出总报告。"""
        if not self._results:
            return
        report = self._builder.build(self._results)
        outputs = self._builder.export(report)
        self.get_logger().info(f'已生成质检报告: {[str(p) for p in outputs]}')
        self._results = []

    def destroy_node(self) -> bool:
        self.flush()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ReportNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
