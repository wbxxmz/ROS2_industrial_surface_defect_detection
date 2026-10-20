"""inputs：输入源实现（阶段 2 实现图片/视频/摄像头）。"""

from inputs.camera_source import CameraSource
from inputs.config import SourceConfig
from inputs.factory import build_source
from inputs.image_source import ImageSource
from inputs.video_source import VideoSource

__all__ = [
    "SourceConfig",
    "ImageSource",
    "VideoSource",
    "CameraSource",
    "build_source",
]
