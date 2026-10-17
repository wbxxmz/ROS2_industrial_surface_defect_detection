"""一键启动：输入节点 + 检测流水线 + 可视化 + 报告（实现文档 阶段9 · 4.2）。"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('defect_bringup'), 'config', 'default.yaml')

    source_type = LaunchConfiguration('source_type')
    source_path = LaunchConfiguration('source_path')
    detector = LaunchConfiguration('detector')

    return LaunchDescription([
        DeclareLaunchArgument('source_type', default_value='camera',
                              description='输入源类型：image / video / camera'),
        DeclareLaunchArgument('source_path', default_value='0',
                              description='图片/视频路径或摄像头设备号 / RTSP 地址'),
        DeclareLaunchArgument('detector', default_value='opencv',
                              description='检测算法：opencv / yolo'),

        Node(
            package='defect_io',
            executable='input_node',
            name='defect_input_node',
            parameters=[config, {'source_type': source_type,
                                 'source_path': source_path}],
            output='screen',
        ),
        Node(
            package='defect_core',
            executable='pipeline_node',
            name='defect_pipeline_node',
            parameters=[config, {'detector': detector}],
            output='screen',
        ),
        Node(
            package='defect_visualization',
            executable='visualize_node',
            name='defect_visualize_node',
            parameters=[config],
            output='screen',
        ),
        Node(
            package='defect_report',
            executable='report_node',
            name='defect_report_node',
            parameters=[config],
            output='screen',
        ),
    ])
