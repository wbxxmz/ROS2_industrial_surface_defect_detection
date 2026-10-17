"""defect_io：图片 / 视频 / 摄像头输入源。"""

from .camera_source import CameraSource
from .image_source import ImageSource
from .input_source import InputSource
from .source_factory import create_input_source
from .video_source import VideoSource

__all__ = [
    'InputSource', 'ImageSource', 'VideoSource', 'CameraSource',
    'create_input_source',
]
