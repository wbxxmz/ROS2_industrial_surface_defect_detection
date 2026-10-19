"""日志系统：分级日志 + 运行日志落盘。

日志级别 / 落盘路径 / 格式 TBD-阶段 2 前定，以下为占位值。
"""

import logging
from pathlib import Path

# TODO(阶段 2 前定)：日志级别、落盘路径、格式
LOG_LEVEL = logging.INFO
LOG_DIR = "logs"
LOG_FILE = "run.log"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

LOGGER_NAME = "defect"


def setup_logging() -> logging.Logger:
    """配置日志（控制台 + 文件落盘），重复调用不重复挂 handler。

    :return: 项目根 logger
    """
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return logger

    logger.setLevel(LOG_LEVEL)
    formatter = logging.Formatter(LOG_FORMAT)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_dir = Path(LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
