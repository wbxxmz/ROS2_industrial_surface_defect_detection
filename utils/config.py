"""统一配置加载（YAML/JSON）。

本阶段只做"原样加载"，不做 schema 校验；
具体字段名与结构 TBD-阶段 2 前定。
"""

import json
from pathlib import Path
from typing import Union

import yaml


def load_config(path: Union[str, Path]) -> dict:
    """按文件后缀加载配置，返回原始 dict。

    :param path: 配置文件路径（.yaml/.yml/.json）
    :return: 配置 dict
    """
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        return json.loads(text) or {}
    return yaml.safe_load(text) or {}
