"""
性能监控工具
"""
import time
import torch
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class PerformanceMetrics:
    """性能指标"""
    generation_times: List[float] = field(default_factory=list)
    memory_usage_mb: List[float] = field(default_factory=list)
    token_counts: List[int] = field(default_factory=list)
    lora_load_times: Dict[str, float] = field(default_factory=dict)
    scene_switch_times: List[float] = field(default_factory=list)


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, enabled: bool = True):
        """
        初始化性能监控器
        
        Args:
            enabled: 是否启用监控
        """
        self.enabled = enabled
        self.metrics = PerformanceMetrics()
        self.start_times: Dict[str, float] = {}
    
    def start_timer(self, name: str):
        """开始计时"""
        if self.enabled:
            self.start_times[name] = time.time()
    
    def end_timer(self, name: str) -> Optional[float]:
        """结束计时并返回耗时"""
        if not self.enabled or name not in self.start_times:
            return None
        
        elapsed = time.time() - self.start_times[name]
        del self.start_times[name]
        return elapsed
    
    def record_generation_time(self, elapsed: float):
        """记录生成时间"""
        if self.enabled:
            self.metrics.generation_times.append(elapsed)
    
    def record_memory_usage(self):
        """记录显存使用"""
        if self.enabled and torch.cuda.is_available():
            memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
            self.metrics.memory_usage_mb.append(memory_mb)
    
    def record_token_count(self, count: int):
        """记录token数量"""
        if self.enabled:
            self.metrics.token_counts.append(count)
    
    def record_lora_load_time(self, role_id: str, elapsed: float):
        """记录LoRA加载时间"""
        if self.enabled:
            self.metrics.lora_load_times[role_id] = elapsed
    
    def record_scene_switch_time(self, elapsed: float):
        """记录场景切换时间"""
        if self.enabled:
            self.metrics.scene_switch_times.append(elapsed)
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        if not self.enabled:
            return {}
        
        stats = {}
        
        # 生成时间统计
        if self.metrics.generation_times:
            stats["generation"] = {
                "count": len(self.metrics.generation_times),
                "avg": sum(self.metrics.generation_times) / len(self.metrics.generation_times),
                "min": min(self.metrics.generation_times),
                "max": max(self.metrics.generation_times),
                "total": sum(self.metrics.generation_times)
            }
        
        # 显存使用统计
        if self.metrics.memory_usage_mb:
            stats["memory"] = {
                "avg_mb": sum(self.metrics.memory_usage_mb) / len(self.metrics.memory_usage_mb),
                "max_mb": max(self.metrics.memory_usage_mb),
                "min_mb": min(self.metrics.memory_usage_mb)
            }
        
        # Token统计
        if self.metrics.token_counts:
            stats["tokens"] = {
                "total": sum(self.metrics.token_counts),
                "avg": sum(self.metrics.token_counts) / len(self.metrics.token_counts),
                "max": max(self.metrics.token_counts)
            }
        
        # LoRA加载时间
        if self.metrics.lora_load_times:
            stats["lora_load"] = {
                "count": len(self.metrics.lora_load_times),
                "total": sum(self.metrics.lora_load_times.values()),
                "avg": sum(self.metrics.lora_load_times.values()) / len(self.metrics.lora_load_times)
            }
        
        # 场景切换时间
        if self.metrics.scene_switch_times:
            stats["scene_switch"] = {
                "count": len(self.metrics.scene_switch_times),
                "total": sum(self.metrics.scene_switch_times),
                "avg": sum(self.metrics.scene_switch_times) / len(self.metrics.scene_switch_times)
            }
        
        return stats
    
    def print_stats(self):
        """打印统计信息"""
        stats = self.get_stats()
        if not stats:
            print("[性能监控] 无统计数据")
            return
        
        print("\n" + "=" * 60)
        print("性能统计")
        print("=" * 60)
        
        if "generation" in stats:
            g = stats["generation"]
            print(f"\n生成统计:")
            print(f"  总次数: {g['count']}")
            print(f"  平均耗时: {g['avg']:.2f}秒")
            print(f"  最短耗时: {g['min']:.2f}秒")
            print(f"  最长耗时: {g['max']:.2f}秒")
            print(f"  总耗时: {g['total']:.2f}秒")
        
        if "memory" in stats:
            m = stats["memory"]
            print(f"\n显存使用:")
            print(f"  平均: {m['avg_mb']:.2f} MB")
            print(f"  最大: {m['max_mb']:.2f} MB")
            print(f"  最小: {m['min_mb']:.2f} MB")
        
        if "tokens" in stats:
            t = stats["tokens"]
            print(f"\nToken统计:")
            print(f"  总计: {t['total']}")
            print(f"  平均: {t['avg']:.0f}")
            print(f"  最大: {t['max']}")
        
        if "lora_load" in stats:
            l = stats["lora_load"]
            print(f"\nLoRA加载:")
            print(f"  次数: {l['count']}")
            print(f"  总耗时: {l['total']:.2f}秒")
            print(f"  平均耗时: {l['avg']:.2f}秒")
        
        if "scene_switch" in stats:
            s = stats["scene_switch"]
            print(f"\n场景切换:")
            print(f"  次数: {s['count']}")
            print(f"  总耗时: {s['total']:.2f}秒")
            print(f"  平均耗时: {s['avg']:.2f}秒")
        
        print("=" * 60 + "\n")

