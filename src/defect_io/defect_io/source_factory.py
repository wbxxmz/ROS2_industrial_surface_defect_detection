"""输入源工厂：由配置决定实例化哪种输入源（实现文档 阶段2 · 4.4）。"""

from .camera_source import CameraSource
from .image_source import ImageSource
from .input_source import InputSource
from .video_source import VideoSource

_SOURCES = {
    'image': ImageSource,
    'video': VideoSource,
    'camera': CameraSource,
}


def create_input_source(source_type: str) -> InputSource:
    """按配置的 input.type 创建输入源，未知类型给出可读提示。"""
    if source_type not in _SOURCES:
        raise KeyError(
            f'未知输入源类型: {source_type}，可选: {list(_SOURCES.keys())}')
    return _SOURCES[source_type]()
