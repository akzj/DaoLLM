"""
配置验证工具
"""
from typing import Dict, List, Optional
from .error_handler import ConfigurationError


def validate_scenario_config(config: Dict) -> List[str]:
    """
    验证剧本配置
    
    Returns:
        错误列表，如果为空则表示配置有效
    """
    errors = []
    
    if "剧本" not in config:
        errors.append("配置中缺少'剧本'字段")
        return errors
    
    scenario = config["剧本"]
    
    # 检查元信息
    if "元信息" not in scenario:
        errors.append("剧本配置中缺少'元信息'字段")
    
    # 检查场景列表
    if "场景列表" not in scenario:
        errors.append("剧本配置中缺少'场景列表'字段")
        return errors
    
    scenes = scenario["场景列表"]
    if not isinstance(scenes, list):
        errors.append("'场景列表'必须是列表类型")
        return errors
    
    if len(scenes) == 0:
        errors.append("'场景列表'不能为空")
    
    # 验证每个场景
    for i, scene in enumerate(scenes):
        if not isinstance(scene, dict):
            errors.append(f"场景列表第{i+1}项必须是字典类型")
            continue
        
        if "场景ID" not in scene:
            errors.append(f"场景列表第{i+1}项缺少'场景ID'字段")
        
        if "节点列表" not in scene:
            errors.append(f"场景'{scene.get('场景ID', '未知')}'缺少'节点列表'字段")
            continue
        
        nodes = scene.get("节点列表", [])
        if not isinstance(nodes, list):
            errors.append(f"场景'{scene.get('场景ID', '未知')}'的'节点列表'必须是列表类型")
            continue
        
        # 验证每个节点
        for j, node in enumerate(nodes):
            if not isinstance(node, dict):
                errors.append(f"场景'{scene.get('场景ID', '未知')}'节点列表第{j+1}项必须是字典类型")
                continue
            
            if "节点ID" not in node:
                errors.append(f"场景'{scene.get('场景ID', '未知')}'节点列表第{j+1}项缺少'节点ID'字段")
    
    return errors


def validate_characters_config(config: Dict) -> List[str]:
    """
    验证角色配置
    
    Returns:
        错误列表，如果为空则表示配置有效
    """
    errors = []
    
    if "角色" not in config:
        errors.append("配置中缺少'角色'字段")
        return errors
    
    roles = config["角色"]
    if not isinstance(roles, list):
        errors.append("'角色'必须是列表类型")
        return errors
    
    if len(roles) == 0:
        errors.append("'角色'列表不能为空")
    
    # 验证每个角色
    role_ids = set()
    for i, role in enumerate(roles):
        if not isinstance(role, dict):
            errors.append(f"角色列表第{i+1}项必须是字典类型")
            continue
        
        required_fields = ["角色ID", "名称", "人设"]
        for field in required_fields:
            if field not in role:
                errors.append(f"角色列表第{i+1}项缺少'{field}'字段")
        
        # 检查角色ID唯一性
        role_id = role.get("角色ID")
        if role_id:
            if role_id in role_ids:
                errors.append(f"角色ID'{role_id}'重复")
            role_ids.add(role_id)
    
    return errors


def validate_configs(scenario_config: Dict, characters_config: Dict) -> None:
    """
    验证配置并抛出异常如果有错误
    
    Args:
        scenario_config: 剧本配置
        characters_config: 角色配置
    
    Raises:
        ConfigurationError: 如果配置有错误
    """
    scenario_errors = validate_scenario_config(scenario_config)
    character_errors = validate_characters_config(characters_config)
    
    all_errors = scenario_errors + character_errors
    
    if all_errors:
        error_msg = "配置验证失败：\n" + "\n".join(f"  - {e}" for e in all_errors)
        raise ConfigurationError(error_msg)

