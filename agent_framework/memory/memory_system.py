"""
记忆系统 - 负责记忆存储、检索和压缩
"""
import time
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import AutoTokenizer


@dataclass
class Memory:
    """记忆数据结构"""
    id: str
    timestamp: float
    type: str  # "角色发言", "玩家选择", "剧情事件"
    content: str
    importance: float = 0.5  # 重要性分数 0-1
    role: Optional[str] = None  # 相关角色
    scene: Optional[str] = None  # 所属场景
    embedding: Optional[np.ndarray] = None  # 语义向量


class MemorySystem:
    """记忆系统 - 混合压缩策略"""
    
    def __init__(self, max_tokens: int = 1024, embedding_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """
        初始化记忆系统
        
        Args:
            max_tokens: 最大token数
            embedding_model_name: embedding模型名称
        """
        self.max_tokens = max_tokens
        self.memories: List[Memory] = []
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B-Chat", trust_remote_code=True)
        self.decay_constant_core = 100  # 核心记忆衰减常数
        self.decay_constant_normal = 10  # 普通记忆衰减常数
        self.decay_constant_low = 1  # 低重要性记忆衰减常数
        self.similarity_threshold = 0.8  # 相似度阈值
    
    def add_memory(
        self,
        content: str,
        memory_type: str,
        role: Optional[str] = None,
        scene: Optional[str] = None,
        importance: Optional[float] = None
    ) -> Memory:
        """
        添加记忆
        
        Args:
            content: 记忆内容
            memory_type: 记忆类型
            role: 相关角色
            scene: 所属场景
            importance: 重要性分数（如果为None则自动计算）
        """
        # 计算重要性分数
        if importance is None:
            importance = self._calculate_importance(content, memory_type, role)
        
        # 生成embedding
        embedding = self.embedding_model.encode(content, convert_to_numpy=True)
        
        # 创建记忆
        memory = Memory(
            id=f"mem_{int(time.time() * 1000)}",
            timestamp=time.time(),
            type=memory_type,
            content=content,
            importance=importance,
            role=role,
            scene=scene,
            embedding=embedding
        )
        
        self.memories.append(memory)
        return memory
    
    def _calculate_importance(self, content: str, memory_type: str, role: Optional[str] = None) -> float:
        """
        计算记忆重要性分数
        
        策略：规则为主，简单快速
        """
        importance = 0.5  # 基础分数
        
        # 剧情关键性（权重40%）
        if memory_type == "剧情事件":
            if "场景切换" in content or "完成" in content:
                importance += 0.4
            elif "节点" in content:
                importance += 0.2
        elif memory_type == "玩家选择":
            importance += 0.2
        elif memory_type == "角色发言":
            importance += 0.1
        
        # 角色相关性（权重30%）
        if role:
            importance += 0.15
        
        # 信息密度（权重30%）
        key_words = ["找到", "获得", "发现", "决定", "重要", "关键"]
        if any(word in content for word in key_words):
            importance += 0.2
        
        return min(1.0, importance)
    
    def compress_memories(self, current_scene: Optional[str] = None) -> List[Memory]:
        """
        压缩记忆 - 混合策略
        
        1. 重要性排序
        2. 语义相似度去重
        3. 时间衰减
        4. 截断到max_tokens
        """
        if not self.memories:
            return []
        
        # 1. 应用时间衰减
        current_time = time.time()
        for memory in self.memories:
            time_diff = current_time - memory.timestamp
            
            # 根据重要性选择衰减常数
            if memory.importance > 0.7:
                decay_constant = self.decay_constant_core
            elif memory.importance > 0.3:
                decay_constant = self.decay_constant_normal
            else:
                decay_constant = self.decay_constant_low
            
            # 计算衰减系数
            decay_factor = 1.0 / (1.0 + time_diff / decay_constant)
            memory.importance *= decay_factor
        
        # 2. 语义相似度去重
        memories_to_keep = []
        for memory in self.memories:
            is_duplicate = False
            for kept_memory in memories_to_keep:
                if memory.embedding is not None and kept_memory.embedding is not None:
                    similarity = np.dot(memory.embedding, kept_memory.embedding) / (
                        np.linalg.norm(memory.embedding) * np.linalg.norm(kept_memory.embedding)
                    )
                    if similarity > self.similarity_threshold:
                        # 保留重要性更高的
                        if memory.importance > kept_memory.importance:
                            memories_to_keep.remove(kept_memory)
                            memories_to_keep.append(memory)
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                memories_to_keep.append(memory)
        
        # 3. 按重要性排序
        memories_to_keep.sort(key=lambda x: x.importance, reverse=True)
        
        # 4. 截断到max_tokens
        compressed_memories = []
        total_tokens = 0
        
        for memory in memories_to_keep:
            tokens = len(self.tokenizer.encode(memory.content))
            if total_tokens + tokens <= self.max_tokens:
                compressed_memories.append(memory)
                total_tokens += tokens
            else:
                break
        
        self.memories = compressed_memories
        return compressed_memories
    
    def get_memories(
        self,
        role: Optional[str] = None,
        scene: Optional[str] = None,
        limit: int = 10
    ) -> List[Memory]:
        """
        检索记忆
        
        Args:
            role: 角色过滤
            scene: 场景过滤
            limit: 返回数量限制
        """
        filtered = self.memories
        
        if role:
            filtered = [m for m in filtered if m.role == role]
        
        if scene:
            filtered = [m for m in filtered if m.scene == scene]
        
        # 按重要性排序
        filtered.sort(key=lambda x: x.importance, reverse=True)
        
        return filtered[:limit]
    
    def get_context_text(self, role: Optional[str] = None, scene: Optional[str] = None) -> str:
        """
        获取上下文文本（用于生成）
        """
        memories = self.get_memories(role=role, scene=scene, limit=20)
        context_parts = []
        
        for memory in memories:
            if memory.type == "角色发言":
                context_parts.append(f"[{memory.role}]: {memory.content}")
            elif memory.type == "玩家选择":
                context_parts.append(f"[玩家]: {memory.content}")
            elif memory.type == "剧情事件":
                context_parts.append(f"[剧情]: {memory.content}")
        
        return "\n".join(context_parts)
    
    def clear_scene_memories(self, scene: str):
        """清除特定场景的记忆（场景切换时调用）"""
        self.memories = [m for m in self.memories if m.scene != scene]
        # 压缩剩余记忆
        self.compress_memories()

