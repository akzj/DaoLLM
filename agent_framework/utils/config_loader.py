"""
配置加载工具
"""
import yaml
from pathlib import Path
from typing import Dict


def load_scenario_config(path: str) -> Dict:
    """加载剧本配置"""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_characters_config(path: str) -> Dict:
    """加载角色配置"""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def validate_config(config: Dict, config_type: str) -> bool:
    """验证配置格式"""
    if config_type == "scenario":
        required_keys = ["剧本"]
        if "剧本" not in config:
            return False
        if "场景列表" not in config["剧本"]:
            return False
        return True
    
    elif config_type == "characters":
        if "角色" not in config:
            return False
        for role in config["角色"]:
            required_keys = ["角色ID", "名称", "人设"]
            if not all(key in role for key in required_keys):
                return False
        return True
    
    return False

