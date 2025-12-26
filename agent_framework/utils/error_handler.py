"""
错误处理工具
"""
import traceback
from typing import Optional, Callable, Any
from functools import wraps


def handle_errors(
    default_return: Any = None,
    log_error: bool = True,
    reraise: bool = False
):
    """
    错误处理装饰器
    
    Args:
        default_return: 发生错误时的默认返回值
        log_error: 是否记录错误日志
        reraise: 是否重新抛出异常
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    print(f"[错误] {func.__name__}: {str(e)}")
                    if reraise:
                        traceback.print_exc()
                if reraise:
                    raise
                return default_return
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    print(f"[错误] {func.__name__}: {str(e)}")
                    if reraise:
                        traceback.print_exc()
                if reraise:
                    raise
                return default_return
        
        # 判断是否是协程函数
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class AgentError(Exception):
    """Agent框架基础异常"""
    pass


class ConfigurationError(AgentError):
    """配置错误"""
    pass


class ModelLoadError(AgentError):
    """模型加载错误"""
    pass


class MemoryError(AgentError):
    """记忆系统错误"""
    pass

