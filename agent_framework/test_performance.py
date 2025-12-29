#!/usr/bin/env python
"""
表演效果测试脚本
用于测试角色表演格式、内容质量等
"""
import asyncio
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agent_framework.main import DramaEngine


async def test_performance():
    """测试表演效果"""
    script_dir = Path(__file__).parent
    scenario_path = str(script_dir / "config" / "test_performance.yaml")
    characters_path = str(script_dir / "config" / "example_characters.yaml")
    
    print("=" * 60)
    print("开始表演效果测试")
    print("=" * 60)
    print(f"场景配置: {scenario_path}")
    print(f"角色配置: {characters_path}")
    print("=" * 60)
    print()
    
    # 创建引擎
    engine = DramaEngine(
        scenario_path=scenario_path,
        characters_path=characters_path,
        base_model_name="Qwen/Qwen1.5-7B-Chat"
    )
    
    # 运行测试
    try:
        await engine.start()
    except KeyboardInterrupt:
        print("\n\n测试中断")
    except Exception as e:
        print(f"\n\n测试错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_performance())

