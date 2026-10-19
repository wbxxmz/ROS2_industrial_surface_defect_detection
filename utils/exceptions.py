"""自定义异常（阶段 0 §5.4 契约规定的 4 类）。

仅在此处定义异常类型；基类方法体不抛这些异常（只抛 NotImplementedError），
由子类实现按契约抛出。

所有自定义异常继承 ProjectError，上层统一捕获 ProjectError 即可。
"""


class ProjectError(Exception):
    """项目内所有自定义异常的基类，供上层统一捕获。"""


class DetectorLoadError(ProjectError):
    """检测器加载/初始化失败（Detector.load）。"""


class DetectorInferenceError(ProjectError):
    """检测器内部推理异常（Detector.detect）。"""


class SourceOpenError(ProjectError):
    """输入源打开失败（InputSource.open）。"""


class InputSourceError(ProjectError):
    """输入源读取运行中断流/出错（InputSource.read）。"""
