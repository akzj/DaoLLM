#!/usr/bin/env python
"""
简单测试脚本 - 测试基本功能
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agent_framework.utils.config_validator import validate_scenario_config, validate_characters_config
from agent_framework.utils.config_loader import load_scenario_config, load_characters_config
import yaml


def test_config_loading():
    """测试配置加载"""
    print("测试配置加载...")
    
    try:
        # 使用相对于脚本文件的路径
        script_dir = Path(__file__).parent
        scenario_path = script_dir / "config" / "example_scenario.yaml"
        characters_path = script_dir / "config" / "example_characters.yaml"
        
        scenario_config = load_scenario_config(str(scenario_path))
        characters_config = load_characters_config(str(characters_path))
        
        print("✓ 配置加载成功")
        
        # 验证配置
        scenario_errors = validate_scenario_config(scenario_config)
        character_errors = validate_characters_config(characters_config)
        
        if scenario_errors:
            print("✗ 剧本配置错误:")
            for error in scenario_errors:
                print(f"  - {error}")
        else:
            print("✓ 剧本配置验证通过")
        
        if character_errors:
            print("✗ 角色配置错误:")
            for error in character_errors:
                print(f"  - {error}")
        else:
            print("✓ 角色配置验证通过")
        
        return len(scenario_errors) == 0 and len(character_errors) == 0
        
    except Exception as e:
        print(f"✗ 配置加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_system():
    """测试记忆系统"""
    print("\n测试记忆系统...")
    
    try:
        from agent_framework.memory.memory_system import MemorySystem
        
        memory = MemorySystem(max_tokens=100)  # 使用较小的token限制便于测试
        
        # 添加一些记忆
        memory.add_memory("李师兄说：我们要小心行事", "角色发言", role="李师兄", importance=0.6)
        memory.add_memory("玩家选择：询问秘境情况", "玩家选择", importance=0.5)
        memory.add_memory("进入场景：客栈休整", "剧情事件", scene="客栈休整", importance=1.0)
        
        print(f"✓ 添加了 {len(memory.memories)} 条记忆")
        
        # 测试检索
        role_memories = memory.get_memories(role="李师兄")
        print(f"✓ 检索到 {len(role_memories)} 条李师兄相关记忆")
        
        # 测试压缩
        compressed = memory.compress_memories()
        print(f"✓ 压缩后保留 {len(compressed)} 条记忆")
        
        # 测试上下文生成
        context = memory.get_context_text()
        print(f"✓ 生成上下文长度: {len(context)} 字符")
        
        return True
        
    except Exception as e:
        print(f"✗ 记忆系统测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_player_input():
    """测试玩家输入（模拟）"""
    print("\n测试玩家输入处理...")
    
    try:
        from agent_framework.utils.player_input import get_player_choice_sync
        
        options = ["选项1", "选项2", "选项3"]
        # 注意：这个测试需要实际输入，所以这里只是测试函数是否可调用
        print("✓ 玩家输入模块加载成功")
        print("  注意：实际输入测试需要交互式环境")
        
        return True
        
    except Exception as e:
        print(f"✗ 玩家输入测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("=" * 60)
    print("AI Agent框架 - 简单测试")
    print("=" * 60)
    
    results = []
    
    # 测试配置加载
    results.append(("配置加载", test_config_loading()))
    
    # 测试记忆系统
    results.append(("记忆系统", test_memory_system()))
    
    # 测试玩家输入
    results.append(("玩家输入", test_player_input()))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n✓ 所有测试通过！")
    else:
        print("\n✗ 部分测试失败，请检查错误信息")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

