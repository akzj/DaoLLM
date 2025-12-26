"""
系统Agent - 负责资源管理、状态管理、场景描述生成
"""
import asyncio
import threading
from typing import Dict, Optional, List
from pathlib import Path
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

import sys
from pathlib import Path
# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from agent_framework.memory.memory_system import MemorySystem


class SystemAgent:
    """系统Agent - 核心资源管理器"""
    
    def __init__(self, base_model_name: str = "Qwen/Qwen1.5-7B-Chat", use_4bit: bool = True):
        """
        初始化系统Agent
        
        Args:
            base_model_name: 基础模型名称
            use_4bit: 是否使用4bit量化
        """
        self.base_model_name = base_model_name
        self.use_4bit = use_4bit
        self.base_model = None
        self.tokenizer = None
        self.lora_lock = threading.Lock()  # LoRA锁
        self.memory_system = MemorySystem()
        self.global_state: Dict = {}
        self.current_scene: Optional[str] = None
        self.loaded_loras: Dict[str, PeftModel] = {}  # 已加载的LoRA
        
        # 初始化模型
        self._load_base_model()
    
    def _load_base_model(self):
        """加载基础模型"""
        print(f"加载基础模型: {self.base_model_name}")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # 配置量化
        if self.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_config = None
        
        # 加载模型
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16 if not self.use_4bit else None,
        )
        
        print("基础模型加载完成")
    
    def load_lora(self, role_id: str, lora_path: str, verbose: bool = True) -> Optional[PeftModel]:
        """
        加载LoRA权重
        
        Args:
            role_id: 角色ID
            lora_path: LoRA路径
            verbose: 是否输出详细信息
        
        Returns:
            LoRA模型，如果路径不存在或加载失败则返回None
        """
        if role_id in self.loaded_loras:
            return self.loaded_loras[role_id]
        
        with self.lora_lock:
            if role_id in self.loaded_loras:
                return self.loaded_loras[role_id]
            
            # 如果路径为空，静默返回None
            if not lora_path or not lora_path.strip():
                return None
            
            lora_path_obj = Path(lora_path)
            
            # 如果路径不存在，静默返回None（使用基础模型）
            if not lora_path_obj.exists():
                return None
            
            # 路径存在，尝试加载
            if verbose:
                print(f"[LoRA] 加载 {role_id} from {lora_path}")
            
            try:
                lora_model = PeftModel.from_pretrained(self.base_model, lora_path)
                self.loaded_loras[role_id] = lora_model
                if verbose:
                    print(f"[LoRA] ✓ {role_id} 加载成功")
                return lora_model
            except Exception as e:
                print(f"[LoRA] ✗ 加载失败 {role_id}: {e}，使用基础模型")
                return None
    
    def unload_lora(self, role_id: str):
        """卸载LoRA权重"""
        if role_id in self.loaded_loras:
            with self.lora_lock:
                if role_id in self.loaded_loras:
                    del self.loaded_loras[role_id]
                    # 清理显存
                    torch.cuda.empty_cache()
                    print(f"卸载LoRA: {role_id}")
    
    def generate_scene_description(self, scene_config: Dict) -> str:
        """
        生成场景描述
        
        Args:
            scene_config: 场景配置
        """
        scene_id = scene_config.get("场景ID", "")
        scene_desc = scene_config.get("场景描述", "")
        
        # 如果有预设描述，直接使用
        if scene_desc:
            return scene_desc
        
        # 否则使用模型生成（Qwen格式）
        prompt = f"<|im_start|>system\n你是一位修仙世界的场景描述者。<|im_end|>\n<|im_start|>user\n请描述修仙世界中的场景：{scene_id}\n场景描述：<|im_end|>\n<|im_start|>assistant\n"
        
        with self.lora_lock:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.base_model.device)
            outputs = self.base_model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
            )
        
        generated = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return generated.strip()
    
    def generate_player_options(self, current_node: Dict, context: str) -> List[str]:
        """
        生成玩家选项
        
        Args:
            current_node: 当前节点配置
            context: 当前上下文
        """
        # 如果节点配置中有预设选项，直接使用
        if "玩家选项" in current_node:
            return current_node["玩家选项"]
        
        # 否则基于上下文生成（Qwen格式）
        system_prompt = "你是一位修仙世界游戏的选项生成者。"
        user_prompt = f"""当前剧情上下文：
{context}

请为玩家生成3-5个选项，每个选项应该：
1. 符合当前剧情
2. 简洁明了（10-20字）
3. 有明确的行动意图

选项："""
        
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        with self.lora_lock:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.base_model.device)
            outputs = self.base_model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.8,
                do_sample=True,
            )
        
        generated = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # 解析生成的选项（支持多种格式）
        options = []
        for line in generated.split("\n"):
            line = line.strip()
            if not line:
                continue
            # 移除编号前缀（如 "1. "、"1、"等）
            for prefix in ["1.", "2.", "3.", "4.", "5.", "1、", "2、", "3、", "4、", "5、", "- "]:
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()
                    break
            if line and len(line) > 2:  # 至少3个字符
                options.append(line)
        
        # 限制数量并返回
        options = options[:5]
        return options if options else ["继续观察", "保持沉默", "询问情况"]
    
    def update_global_state(self, key: str, value):
        """更新全局状态"""
        self.global_state[key] = value
    
    def get_global_state(self, key: str, default=None):
        """获取全局状态"""
        return self.global_state.get(key, default)

