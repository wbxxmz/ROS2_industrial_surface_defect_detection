"""阶段 2 输入源工厂与契约一致性测试（§4.4 / §5）。"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.input_source import InputSource  # noqa: E402
from core.types import Frame  # noqa: E402
from inputs import CameraSource, ImageSource, VideoSource, build_source  # noqa: E402
from inputs.config import SourceConfig  # noqa: E402


def test_factory_returns_expected_subclasses():
    assert isinstance(build_source({"type": "image", "path": "a.png"}), ImageSource)
    assert isinstance(build_source({"type": "video", "path": "a.mp4"}), VideoSource)
    assert isinstance(build_source({"type": "camera", "device_id": 0}), CameraSource)


def test_factory_returns_unopened_instance():
    """工厂返回未 open 的实例，调用方需自行 open。"""
    source = build_source({"type": "image", "path": "a.png"})
    assert isinstance(source, InputSource)


def test_factory_unknown_type_raises_value_error():
    with pytest.raises(ValueError):
        build_source({"type": "webcam", "path": "a.png"})


def test_all_sources_inherit_input_source():
    for cls in (ImageSource, VideoSource, CameraSource):
        assert issubclass(cls, InputSource)
        assert isinstance(cls(), InputSource)


def test_contract_method_signatures():
    """签名与阶段 0 §5.4.3 逐字一致：方法名、参数名。"""
    import inspect

    for cls in (ImageSource, VideoSource, CameraSource):
        assert list(inspect.signature(cls.open).parameters) == ["self", "source_cfg"]
        assert list(inspect.signature(cls.read).parameters) == ["self"]
        assert list(inspect.signature(cls.close).parameters) == ["self"]


def test_frame_fields_match_frozen_contract():
    """Frame 字段名/顺序与阶段 0 §5.4.4 冻结表逐字一致。"""
    import dataclasses

    assert [f.name for f in dataclasses.fields(Frame)] == [
        "image",
        "width",
        "height",
        "frame_index",
        "timestamp",
    ]


def test_source_config_defaults_match_doc():
    """source_cfg 默认值与阶段 2 §4.4 一致。"""
    cfg = SourceConfig.from_dict({"type": "camera", "device_id": 0})
    assert cfg.queue_size == 8
    assert cfg.timeout == 5.0
    assert cfg.reconnect is True
    assert cfg.reconnect_interval == 2.0
    assert cfg.reconnect_max == 5
    assert cfg.step == 1
    assert cfg.loop is False
    assert cfg.extensions == [".jpg", ".jpeg", ".png", ".bmp"]


def test_frame_is_dataclass_with_expected_types():
    frame = Frame(
        image=np.zeros((2, 2, 3), dtype=np.uint8),
        width=2,
        height=2,
        frame_index=0,
        timestamp=0.0,
    )
    assert isinstance(frame.width, int)
    assert isinstance(frame.frame_index, int)
    assert isinstance(frame.timestamp, float)
