"""
角色Agent - 负责角色发言生成
"""
import asyncio
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor
import sys
from pathlib import Path
# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from agent_framework.agents.system_agent import SystemAgent


class RoleAgent:
    """角色Agent - 角色发言生成器"""
    
    def __init__(self, system_agent: SystemAgent, role_config: Dict):
        """
        初始化角色Agent
        
        Args:
            system_agent: 系统Agent实例
            role_config: 角色配置
        """
        self.system_agent = system_agent
        self.role_id = role_config["角色ID"]
        self.role_name = role_config.get("名称", self.role_id)
        self.persona = role_config.get("人设", "")
        self.lora_path = role_config.get("LoRA路径", "")
        self.initial_goal = role_config.get("初始目标", "")
        self.lora_model = None
        self.executor = ThreadPoolExecutor(max_workers=2)  # 线程池
    
    async def generate_speech(
        self,
        context: str,
        current_node: Dict,
        memory_context: str = ""
    ) -> Dict[str, str]:
        """
        生成角色发言（异步）
        
        Args:
            context: 当前剧情上下文
            current_node: 当前节点配置
            memory_context: 记忆上下文
        
        Returns:
            {"dialogue": "对话内容", "description": "行为描述"}
        """
        # 加载LoRA（如果需要）
        if self.lora_path and not self.lora_model:
            self.lora_model = self.system_agent.load_lora(self.role_id, self.lora_path, verbose=False)
        
        # 调试信息：显示使用的模型
        model_type = "LoRA微调模型" if self.lora_model else "基础模型"
        print(f"[角色-{self.role_name}] 使用{model_type}生成发言")
        
        # 构建Prompt
        prompt = self._build_prompt(context, current_node, memory_context)
        
        # 在线程池中执行生成（避免阻塞）
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self._generate_with_lock,
            prompt
        )
        
        return result
    
    def _build_prompt(
        self,
        context: str,
        current_node: Dict,
        memory_context: str
    ) -> str:
        """构建生成Prompt（Qwen格式）"""
        # 构建系统提示
        system_prompt = f"""你是{self.role_name}，一位修仙世界的角色。

角色人设：
{self.persona}

角色目标：{self.initial_goal}"""

        # 构建用户提示
        user_prompt = f"""当前剧情：
{context}

相关记忆：
{memory_context if memory_context else "无"}

请生成你的发言，要求：
1. 对话内容（50-200字，符合角色人设和当前剧情，使用第一人称）
2. 行为描述（10-30字，第三人称，舞台剧风格，描述你的动作、神态）

输出格式：
对话：[你的对话内容]
描述：（你的行为描述）

对话："""

        # Qwen格式：<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        return prompt
    
    def _generate_with_lock(self, prompt: str) -> Dict[str, str]:
        """
        使用LoRA锁生成（同步方法，在线程池中执行）
        
        Args:
            prompt: 生成Prompt
        
        Returns:
            {"dialogue": "对话", "description": "描述"}
        """
        with self.system_agent.lora_lock:
            # 选择模型（如果有LoRA则使用LoRA模型）
            model = self.lora_model if self.lora_model else self.system_agent.base_model
            
            inputs = self.system_agent.tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
            )
        
        generated = self.system_agent.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        # 解析生成结果
        return self._parse_generation(generated)
    
    def _parse_generation(self, generated: str) -> Dict[str, str]:
        """解析生成结果"""
        dialogue = ""
        description = ""
        
        # 简单解析（实际应该更robust）
        lines = generated.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("对话：") or (not dialogue and line):
                dialogue = line.replace("对话：", "").strip()
            elif line.startswith("描述：") or line.startswith("（"):
                description = line.replace("描述：", "").strip()
                if description.startswith("（") and description.endswith("）"):
                    description = description[1:-1]
        
        # 如果没有解析到，使用简单策略
        if not dialogue:
            dialogue = generated[:200].strip()
        if not description:
            description = "（看向众人）"
        
        return {
            "dialogue": dialogue,
            "description": description
        }
    
    def cleanup(self):
        """清理资源"""
        self.executor.shutdown(wait=False)

