# 算法注册表：实现"算法可替换"的核心机制（对应实现文档 阶段0 / 阶段1）

from typing import Callable, Dict, List

from .detector_base import Detector

_DETECTORS: Dict[str, Callable[[], Detector]] = {}


def register_detector(name: str) -> Callable[[Callable[[], Detector]], Callable[[], Detector]]:
    """装饰器：把一个检测器工厂函数注册到全局注册表。"""

    def _wrap(factory: Callable[[], Detector]) -> Callable[[], Detector]:
        _DETECTORS[name] = factory
        return factory

    return _wrap


def create_detector(name: str) -> Detector:
    """按名称创建检测器实例；名称不存在时抛出可读异常。"""
    if name not in _DETECTORS:
        raise KeyError(
            f'未注册的检测算法: {name}，已注册: {list(_DETECTORS.keys())}')
    return _DETECTORS[name]()


def available_detectors() -> List[str]:
    """返回已注册算法名称列表，供界面下拉选择。"""
    return sorted(_DETECTORS.keys())
