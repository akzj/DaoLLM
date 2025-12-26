#!/usr/bin/env python
"""
运行脚本 - 简化启动流程
"""
import asyncio
import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agent_framework.main import DramaEngine


def main():
    parser = argparse.ArgumentParser(description="修仙世界AI Agent框架")
    # 默认路径相对于脚本文件
    script_dir = Path(__file__).parent
    default_scenario = str(script_dir / "config" / "example_scenario.yaml")
    default_characters = str(script_dir / "config" / "example_characters.yaml")
    
    parser.add_argument(
        "--scenario",
        type=str,
        default=default_scenario,
        help="剧本配置文件路径"
    )
    parser.add_argument(
        "--characters",
        type=str,
        default=default_characters,
        help="角色配置文件路径"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen1.5-7B-Chat",
        help="基础模型名称或路径"
    )
    
    args = parser.parse_args()
    
    # 检查配置文件是否存在
    scenario_path = Path(args.scenario)
    characters_path = Path(args.characters)
    
    if not scenario_path.exists():
        print(f"错误: 剧本配置文件不存在: {scenario_path}")
        return
    
    if not characters_path.exists():
        print(f"错误: 角色配置文件不存在: {characters_path}")
        return
    
    # 创建引擎并运行
    engine = DramaEngine(
        scenario_path=str(scenario_path.absolute()),
        characters_path=str(characters_path.absolute()),
        base_model_name=args.model
    )
    
    try:
        asyncio.run(engine.start())
    except KeyboardInterrupt:
        print("\n\n用户中断，退出程序")
    except Exception as e:
        print(f"\n\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

