"""阶段 2 §4.4：输入源工厂（按 type 实例化对应子类）。

按配置创建 ``ImageSource / VideoSource / CameraSource``；
注意返回的是**未 open** 的实例，调用方需自行 ``open(source_cfg)``。
未知 type 或非法参数由 ``SourceConfig.from_dict`` 抛 ValueError。
"""

from typing import Dict, Type

from core.input_source import InputSource
from inputs.camera_source import CameraSource
from inputs.config import SourceConfig
from inputs.image_source import ImageSource
from inputs.video_source import VideoSource

SOURCES: Dict[str, Type[InputSource]] = {
    "image": ImageSource,
    "video": VideoSource,
    "camera": CameraSource,
}


def build_source(source_cfg: dict) -> InputSource:
    """按 ``input.type`` 实例化输入源（未 open）。"""
    cfg = SourceConfig.from_dict(source_cfg)
    return SOURCES[cfg.type]()
