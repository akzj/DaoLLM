"""
玩家输入处理工具
"""
import asyncio
import sys
from typing import List, Optional


async def get_player_choice_async(options: List[str], timeout: float = 300.0) -> Optional[str]:
    """
    异步获取玩家选择
    
    Args:
        options: 选项列表
        timeout: 超时时间（秒），默认5分钟
    
    Returns:
        玩家选择的选项文本，如果超时或取消则返回None
    """
    print("\n[玩家选项]")
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    print("请输入选项编号（1-{}）或直接输入选项内容：".format(len(options)))
    
    # 使用asyncio创建输入任务
    loop = asyncio.get_event_loop()
    
    try:
        # 在单独的线程中等待输入，避免阻塞事件循环
        def _get_input():
            try:
                return input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                return None
        
        # 最多重试3次
        max_retries = 3
        for retry in range(max_retries):
            user_input = await asyncio.wait_for(
                loop.run_in_executor(None, _get_input),
                timeout=timeout
            )
            
            if not user_input:
                if retry < max_retries - 1:
                    print("[提示] 请输入选项编号或选项内容")
                    continue
                return None
            
            user_input = user_input.strip()
            
            # 过滤掉常见的命令提示符和shell命令
            if any(cmd in user_input for cmd in ["./", "python", "run.py", "zsh:", "bash:"]):
                if retry < max_retries - 1:
                    print(f"[警告] 检测到可能的命令输入，请重新选择（剩余重试次数: {max_retries - retry - 1}）")
                    continue
                else:
                    print("[警告] 输入无效，使用默认选项")
                    return options[0] if options else None
            
            # 尝试解析为数字
            try:
                choice_idx = int(user_input) - 1
                if 0 <= choice_idx < len(options):
                    return options[choice_idx]
            except ValueError:
                pass
            
            # 尝试精确匹配选项文本
            for option in options:
                if user_input == option:
                    return option
            
            # 尝试部分匹配
            for option in options:
                if user_input in option or option in user_input:
                    return option
            
            # 如果都不匹配，提示重试
            if retry < max_retries - 1:
                print(f"[提示] 未识别的输入，请重新选择（剩余重试次数: {max_retries - retry - 1}）")
                continue
            else:
                # 最后一次重试失败，使用默认选项
                print("[警告] 输入无效，使用默认选项")
                return options[0] if options else None
        
        return None
        
    except asyncio.TimeoutError:
        print("\n[超时] 未在规定时间内做出选择，使用默认选项")
        return options[0] if options else None
    except (EOFError, KeyboardInterrupt):
        print("\n[取消] 玩家取消选择")
        return None


def get_player_choice_sync(options: List[str]) -> Optional[str]:
    """
    同步获取玩家选择（用于测试或非异步环境）
    
    Args:
        options: 选项列表
    
    Returns:
        玩家选择的选项文本
    """
    print("\n[玩家选项]")
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    print("请输入选项编号（1-{}）或直接输入选项内容：".format(len(options)))
    
    try:
        user_input = input("> ").strip()
        
        # 尝试解析为数字
        try:
            choice_idx = int(user_input) - 1
            if 0 <= choice_idx < len(options):
                return options[choice_idx]
        except ValueError:
            pass
        
        # 尝试匹配选项文本
        for option in options:
            if user_input in option or option in user_input:
                return option
        
        # 如果都不匹配，返回用户输入
        if user_input:
            return user_input
        
        return options[0] if options else None
        
    except (EOFError, KeyboardInterrupt):
        print("\n[取消] 玩家取消选择")
        return None

