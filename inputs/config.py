"""阶段 2 §4.4：输入源配置（`@dataclass` 约束，不用裸 dict）。

`open(source_cfg: dict)` 的入参仍是 dict（阶段 0 §5.4.3 契约），
进入子类后立即用本模块的 `SourceConfig.from_dict()` 校验并转成强类型对象。
字段名与默认值见阶段 2 §4.4；取值非法（如 ≤0 的 step / queue_size）抛 ValueError。
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

DEFAULT_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]

#: §4.4 冻结的字段全集；出现其它字段说明拼写有误，直接报错
FIELDS = {
    "type",
    "path",
    "dir",
    "extensions",
    "step",
    "loop",
    "start_frame",
    "end_frame",
    "device_id",
    "url",
    "queue_size",
    "timeout",
    "reconnect",
    "reconnect_interval",
    "reconnect_max",
}

SOURCE_TYPES = ("image", "video", "camera")


def _check_int(name: str, value: Any, minimum: int) -> int:
    """整型校验：拒绝 bool（bool 是 int 子类）与越界值。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} 必须是 int，收到 {value!r}")
    if value < minimum:
        raise ValueError(f"{name} 必须 >= {minimum}，收到 {value}")
    return value


def _check_num(name: str, value: Any, minimum: float, strict: bool = False) -> float:
    """浮点校验：strict=True 时取严格大于 minimum。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是数值，收到 {value!r}")
    if (value <= minimum) if strict else (value < minimum):
        raise ValueError(f"{name} 必须 {'>' if strict else '>='} {minimum}，收到 {value}")
    return float(value)


def _check_optional_int(name: str, value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    return _check_int(name, value, 0)


@dataclass
class SourceConfig:
    """输入源配置（阶段 2 §4.4 字段表）。"""

    type: str
    path: Optional[str] = None
    dir: Optional[str] = None
    extensions: List[str] = field(default_factory=lambda: list(DEFAULT_EXTENSIONS))
    step: int = 1
    loop: bool = False
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    device_id: Optional[int] = None
    url: Optional[str] = None
    queue_size: int = 8
    timeout: float = 5.0
    reconnect: bool = True
    reconnect_interval: float = 2.0
    reconnect_max: int = 5

    @classmethod
    def from_dict(cls, cfg: dict) -> "SourceConfig":
        """按 §4.4 校验并构造；非法参数抛 ValueError。"""
        if not isinstance(cfg, dict):
            raise ValueError(f"source_cfg 必须是 dict，收到 {type(cfg).__name__}")
        if "type" not in cfg:
            raise ValueError("source_cfg 缺少必填字段 type")
        unknown = set(cfg) - FIELDS
        if unknown:
            raise ValueError(f"source_cfg 含未知字段：{sorted(unknown)}；允许字段见阶段 2 §4.4")
        config = cls(**cfg)
        config.validate()
        return config

    def validate(self) -> None:
        """逐项校验；任一不合法即抛 ValueError。"""
        if self.type not in SOURCE_TYPES:
            raise ValueError(f"type 必须是 {SOURCE_TYPES} 之一，收到 {self.type!r}")

        _check_int("step", self.step, 1)
        _check_int("queue_size", self.queue_size, 1)
        _check_int("reconnect_max", self.reconnect_max, 0)
        _check_num("timeout", self.timeout, 0, strict=True)
        _check_num("reconnect_interval", self.reconnect_interval, 0, strict=True)
        self.start_frame = _check_optional_int("start_frame", self.start_frame)
        self.end_frame = _check_optional_int("end_frame", self.end_frame)
        if (
            self.start_frame is not None
            and self.end_frame is not None
            and self.start_frame > self.end_frame
        ):
            raise ValueError(f"start_frame 不得大于 end_frame：{self.start_frame} > {self.end_frame}")

        if not isinstance(self.loop, bool):
            raise ValueError(f"loop 必须是 bool，收到 {self.loop!r}")
        if not isinstance(self.reconnect, bool):
            raise ValueError(f"reconnect 必须是 bool，收到 {self.reconnect!r}")

        if not isinstance(self.extensions, (list, tuple)) or not self.extensions:
            raise ValueError(f"extensions 必须是非空列表，收到 {self.extensions!r}")
        if not all(isinstance(ext, str) and ext.startswith(".") for ext in self.extensions):
            raise ValueError(f"extensions 必须是以 '.' 开头的字符串列表，收到 {self.extensions!r}")

        self._validate_required_by_type()

    def _validate_required_by_type(self) -> None:
        """按 type 校验必备字段（§4.4「适用源」列）。"""
        if self.type == "image":
            if (self.path is None) == (self.dir is None):
                raise ValueError("image 源必须且只能给出 path 或 dir 之一")
        elif self.type == "video":
            if not self.path:
                raise ValueError("video 源必须给出 path")
        elif self.type == "camera":
            if (self.device_id is None) == (self.url is None):
                raise ValueError("camera 源必须且只能给出 device_id 或 url 之一")
