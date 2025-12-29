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
        self.acting_guidance: Optional[Dict] = None  # 导演的表演指导
    
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
    
    def set_acting_guidance(self, guidance: Dict):
        """设置导演的表演指导"""
        self.acting_guidance = guidance
    
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

        # 如果有导演的表演指导，添加到提示中
        guidance_text = ""
        if self.acting_guidance and self.acting_guidance.get("role") == self.role_id:
            guidance = self.acting_guidance
            guidance_text = f"""

【导演指导】
表演要求：{guidance.get("guidance", "")}
情绪状态：{guidance.get("emotion", "")}
关注点：{guidance.get("focus", "")}
请严格按照导演指导进行表演。"""
            # 使用后清除指导（一次性使用）
            self.acting_guidance = None

        # 构建用户提示
        user_prompt = f"""当前剧情：
{context}

相关记忆：
{memory_context if memory_context else "无"}
{guidance_text}

请生成你的发言，要求：
1. **对话内容**（50-150字，符合角色人设和当前剧情，使用第一人称，避免重复之前的内容）
   - ⚠️ 对话内容必须是纯对话，不要包含任何动作描述
   - ⚠️ 不要使用"我说："、"我说道："、"我询问："、"我对XX说："等格式
   - ⚠️ 直接输出对话内容，例如："师兄，这次试炼咱们准备好了吗？"而不是"我对师兄说："师兄，这次试炼咱们准备好了吗？""
2. **行为描述**（10-25字，第三人称，舞台剧风格，简洁描述动作、神态，不要重复对话内容）

⚠️ **严格禁止**：
- ❌ 不要输出任何指导性文本（如"接下来"、"现在开始"、"按照导演指导"等）
- ❌ 不要输出格式说明（如"以上内容"、"描述："、"对话："等标记）
- ❌ 不要输出示例性文本（如"例如"、"比如"等）
- ❌ 对话和描述必须分开，不要混在一起
- ❌ 避免重复之前发言的内容
- ❌ **对话中不要包含动作描述**（如"我紧紧握拳，对XX说："、"我深吸一口气，低声说："等）
- ❌ **对话中不要包含角色名+动作**（如"张师弟看着XX说："、"李师兄对XX说："等）
- ❌ **对话中不要使用第二人称动作描述**（如"你询问XX："、"你看着XX说："等）

✅ **必须做到**：
- 只输出对话内容和行为描述
- 保持角色个性，让对话生动自然
- 描述简洁，对话自然
- **对话内容必须是纯对话，动作描述放在"描述"部分**

输出格式（严格遵循，只输出这两行）：
对话：[你的对话内容，纯对话，不要包含动作描述]
描述：（你的行为描述）

现在开始生成："""

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
        """解析生成结果（改进版 - 更智能的格式识别和清理）"""
        import re
        
        dialogue = ""
        description = ""
        
        # 清理输入：移除多余的标记
        generated = generated.strip()
        
        # 方法1：尝试匹配标准格式 "对话：...描述：（...）"
        pattern1 = r'对话[：:]\s*(.+?)(?:描述[：:]|（|\(|$)'
        match1 = re.search(pattern1, generated, re.DOTALL)
        if match1:
            dialogue = match1.group(1).strip()
        
        pattern2 = r'描述[：:]?\s*[（(](.+?)[）)]'
        match2 = re.search(pattern2, generated, re.DOTALL)
        if match2:
            description = match2.group(1).strip()
        
        # 方法2：如果没有找到标准格式，尝试按行解析
        if not dialogue or not description:
            lines = [l.strip() for l in generated.split("\n") if l.strip()]
            current_section = None
            
            for line in lines:
                # 检测章节标记
                if re.match(r'^对话[：:]?\s*', line):
                    current_section = "dialogue"
                    dialogue = re.sub(r'^对话[：:]?\s*', '', line).strip()
                elif re.match(r'^描述[：:]?\s*', line):
                    current_section = "description"
                    desc_text = re.sub(r'^描述[：:]?\s*', '', line).strip()
                    # 移除括号
                    desc_text = re.sub(r'^[（(](.+?)[）)]$', r'\1', desc_text)
                    description = desc_text
                elif re.match(r'^[（(].+?[）)]$', line) and current_section != "dialogue":
                    # 单独一行的括号内容，可能是描述
                    description = re.sub(r'^[（(](.+?)[）)]$', r'\1', line).strip()
                elif current_section == "dialogue":
                    # 继续收集对话内容
                    if dialogue:
                        dialogue += " " + line
                    else:
                        dialogue = line
                elif current_section == "description":
                    # 继续收集描述内容
                    if description:
                        description += " " + line
                    else:
                        description = line
                elif not dialogue and not description:
                    # 智能识别：长文本通常是对话，短文本可能是描述
                    if len(line) > 40 or ("，" in line and "。" in line):
                        dialogue = line
                    elif len(line) < 30 and not re.match(r'^[（(]', line):
                        description = line
        
        # 方法3：如果还是没有，尝试提取引号内容作为对话
        if not dialogue:
            quote_pattern = r'["""](.+?)["""]'
            quote_match = re.search(quote_pattern, generated)
            if quote_match:
                dialogue = quote_match.group(1).strip()
        
        # 清理格式
        dialogue = dialogue.strip()
        description = description.strip()
        
        # 移除所有格式标记
        dialogue = re.sub(r'^(对话[：:]?\s*)', '', dialogue)
        dialogue = re.sub(r'^(描述[：:]?\s*)', '', dialogue)
        description = re.sub(r'^(对话[：:]?\s*)', '', description)
        description = re.sub(r'^(描述[：:]?\s*)', '', description)
        
        # 移除多余的括号和标记
        description = re.sub(r'^[（(](.+?)[）)]$', r'\1', description)
        description = re.sub(r'^描述[：:]?\s*', '', description)
        
        # 移除"描述："标记如果出现在对话中
        dialogue = re.sub(r'描述[：:]?\s*[（(](.+?)[）)]', '', dialogue)
        
        # 清理多余的空白和标点
        dialogue = re.sub(r'\s+', ' ', dialogue).strip()
        description = re.sub(r'\s+', ' ', description).strip()
        
        # Fallback策略：如果还是没有解析到
        if not dialogue:
            # 尝试提取第一段较长的文本作为对话
            lines = [l.strip() for l in generated.split("\n") if l.strip()]
            for line in lines:
                line_clean = re.sub(r'^[（(].+?[）)]$', '', line).strip()
                if len(line_clean) > 20 and not line_clean.startswith("描述") and not line_clean.startswith("对话"):
                    dialogue = line_clean
                    break
            if not dialogue:
                # 最后尝试：取前200字符
                dialogue = generated[:200].strip()
                # 移除明显的格式标记
                dialogue = re.sub(r'^(对话[：:]?\s*|描述[：:]?\s*)', '', dialogue)
        
        # 移除指导性文本（不应该出现在输出中）
        guidance_patterns = [
            r'接下来[，,。.]?\s*',
            r'现在开始[表演表演]?[：:：]?\s*',
            r'按照导演指导[，,。.]?\s*',
            r'以上内容[，,。.]?\s*',
            r'例如[，,。.]?\s*',
            r'比如[，,。.]?\s*',
            r'描述[：:]?\s*',
            r'对话[：:]?\s*',
            r'接下来[，,。.]?\s*我会继续生成[，,。.]?\s*',
            r'请根据[上述]?要求继续生成[，,。.]?\s*',
            r'通过这段对话和[，,。.]?\s*',
            r'接下来的演出[，,。.]?\s*',
            r'现在开始表演[：:：]?\s*',
            r'生成[：:]?\s*',  # 清理"生成："标记
            r'^生成[：:]?\s*$',  # 清理单独的"生成："行
            r'演[：:]?\s*',  # 清理"演："标记
            r'演员名[）)]?\s*',  # 清理"演员名"等
            r'缓慢地停顿[，,。.]?\s*',
            r'眼神坚定地看着镜头[，,。.]?\s*',
            r'然后继续[以].*?说出下文[。.]?\s*',
        ]
        
        for pattern in guidance_patterns:
            dialogue = re.sub(pattern, '', dialogue, flags=re.IGNORECASE)
            description = re.sub(pattern, '', description, flags=re.IGNORECASE)
        
        # 移除包含指导性文本的整行
        dialogue_lines = dialogue.split('\n')
        dialogue = '\n'.join([line for line in dialogue_lines 
                             if not re.search(r'(接下来|现在开始|按照导演|以上内容|例如|比如|描述[：:]|对话[：:]|演[：:]|演员名|生成[：:])', line, re.IGNORECASE)])
        dialogue = re.sub(r'\s+', ' ', dialogue).strip()
        
        description_lines = description.split('\n')
        description = '\n'.join([line for line in description_lines 
                                if not re.search(r'(接下来|现在开始|按照导演|以上内容|例如|比如|描述[：:]|对话[：:]|演[：:]|演员名|生成[：:]|缓慢地停顿|眼神坚定地看着镜头|然后继续)', line, re.IGNORECASE)])
        description = re.sub(r'\s+', ' ', description).strip()
        
        # 检测并修复角色混淆问题
        # 1. 如果描述中提到了其他角色名（不是当前角色），清理或修正
        if description:
            # 检查描述是否以其他角色名开头（常见错误）
            other_role_patterns = [
                r'^(李师兄|张师弟|小师妹|王师兄|赵师妹)',
            ]
            for pattern in other_role_patterns:
                if re.match(pattern, description):
                    # 如果描述以其他角色名开头，移除该角色名
                    description = re.sub(pattern, '', description).strip()
                    # 如果移除后描述为空或太短，使用默认描述
                    if len(description) < 5:
                        description = ""
                    break
        
        # 2. 清理对话中的角色名引用（常见错误：对话中包含"角色名+动作"的描述）
        if dialogue:
            # 检查对话是否以其他角色名开头（常见错误）
            dialogue_start_pattern = r'^["""]?(李师兄|张师弟|小师妹|王师兄|赵师妹)[：:，,]\s*'
            if re.match(dialogue_start_pattern, dialogue):
                # 移除开头的角色名
                dialogue = re.sub(dialogue_start_pattern, '', dialogue).strip()
                # 移除多余的引号
                dialogue = re.sub(r'^["""](.+?)["""]$', r'\1', dialogue)
            
            # 清理对话中引号内的角色名+动作描述+冒号+引号+实际对话
            # 模式1: "角色名动作描述："实际对话"
            # 例如: "张师弟看着李师兄，眉头微蹙，语气中透着好奇："李师兄，这次试炼究竟是什么来头？规矩严不严？""
            # 需要匹配到最后一个引号对
            quote_role_action_pattern1 = r'["""](李师兄|张师弟|小师妹|王师兄|赵师妹)[^"""]*?[：:]["""](.+?)["""]\s*$'
            match = re.search(quote_role_action_pattern1, dialogue)
            if match:
                # 提取实际对话内容（最后一个引号对内的内容）
                actual_dialogue = match.group(2).strip()
                # 替换整个匹配部分为实际对话
                dialogue = re.sub(quote_role_action_pattern1, actual_dialogue, dialogue, count=1)
            
            # 模式2: "角色名动作描述，"实际对话"
            # 例如: "张师弟坚定地看着小师妹，轻拍她的肩膀说，"别担心，有我在呢。咱们分工明确，一起闯过难关！""
            simple_role_action_pattern = r'["""](李师兄|张师弟|小师妹|王师兄|赵师妹)[^"""]*?[，,]["""](.+?)["""]\s*$'
            match = re.search(simple_role_action_pattern, dialogue)
            if match:
                # 提取实际对话内容
                actual_dialogue = match.group(2).strip()
                # 替换整个匹配部分为实际对话
                dialogue = re.sub(simple_role_action_pattern, actual_dialogue, dialogue, count=1)
            
            # 模式3: 角色名+动作描述+冒号+引号+实际对话（没有外层引号）
            # 例如: 张师弟看着李师兄，眉头微蹙，语气中透着好奇："李师兄，这次试炼究竟是什么来头？规矩严不严？"
            no_outer_quote_pattern = r'^(李师兄|张师弟|小师妹|王师兄|赵师妹)[^"""]*?[：:]["""](.+?)["""]\s*$'
            match = re.search(no_outer_quote_pattern, dialogue)
            if match:
                actual_dialogue = match.group(2).strip()
                dialogue = re.sub(no_outer_quote_pattern, actual_dialogue, dialogue, count=1)
            
            # 模式4: 角色名+动作描述+逗号+引号+实际对话（没有外层引号）
            # 例如: 张师弟坚定地看着小师妹，轻拍她的肩膀说，"别担心，有我在呢。"
            no_outer_quote_comma_pattern = r'^(李师兄|张师弟|小师妹|王师兄|赵师妹)[^"""]*?[，,]["""](.+?)["""]\s*$'
            match = re.search(no_outer_quote_comma_pattern, dialogue)
            if match:
                actual_dialogue = match.group(2).strip()
                dialogue = re.sub(no_outer_quote_comma_pattern, actual_dialogue, dialogue, count=1)
            
            # 模式5: "我"开头的动作描述+冒号+引号+实际对话
            # 例如: 我深吸一口气，看着张师弟和小师妹，坚定地说："我们要以团队为先..."
            i_action_colon_pattern = r'^我[^"""]*?[：:]["""](.+?)["""]\s*$'
            match = re.search(i_action_colon_pattern, dialogue)
            if match:
                actual_dialogue = match.group(1).strip()
                dialogue = re.sub(i_action_colon_pattern, actual_dialogue, dialogue, count=1)
            
            # 模式6: "我"开头的动作描述+逗号+引号+实际对话
            # 例如: 我深吸一口气，看着张师弟和小师妹，坚定地说，"我们要以团队为先..."
            i_action_comma_pattern = r'^我[^"""]*?[，,]["""](.+?)["""]\s*$'
            match = re.search(i_action_comma_pattern, dialogue)
            if match:
                actual_dialogue = match.group(1).strip()
                dialogue = re.sub(i_action_comma_pattern, actual_dialogue, dialogue, count=1)
            
            # 模式7: 对话中包含"我+动作+说："的模式（在引号内）
            # 例如: "我紧紧握拳，对李师兄说："师兄，这次试炼咱们准备好了吗？""
            i_action_say_pattern = r'["""]我[^"""]*?[，,，]?[对向]?[^"""]*?[说讲道][：:]["""](.+?)["""]\s*$'
            match = re.search(i_action_say_pattern, dialogue)
            if match:
                actual_dialogue = match.group(1).strip()
                dialogue = re.sub(i_action_say_pattern, actual_dialogue, dialogue, count=1)
            
            # 模式8: 对话中包含"我+动作+说："的模式（无外层引号）
            # 例如: 我紧紧握拳，对李师兄说："师兄，这次试炼咱们准备好了吗？"
            i_action_say_no_quote_pattern = r'^我[^"""]*?[，,，]?[对向]?[^"""]*?[说讲道][：:]["""](.+?)["""]\s*$'
            match = re.search(i_action_say_no_quote_pattern, dialogue)
            if match:
                actual_dialogue = match.group(1).strip()
                dialogue = re.sub(i_action_say_no_quote_pattern, actual_dialogue, dialogue, count=1)
            
            # 模式9: 对话中包含"我对XX说："的模式
            # 例如: "我对张师弟说："张师弟，你对试炼的理解已经让我放心...""
            i_to_role_say_pattern = r'["""]?我[对向]([^"""]+?)[说讲道][：:]["""](.+?)["""]\s*$'
            match = re.search(i_to_role_say_pattern, dialogue)
            if match:
                actual_dialogue = match.group(2).strip()
                dialogue = re.sub(i_to_role_say_pattern, actual_dialogue, dialogue, count=1)
            
            # 模式10: 对话中包含"我+动作+询问/问："的模式
            # 例如: "面对即将到来的试炼，我询问张师弟："准备好应对了吗？""
            i_action_ask_pattern = r'["""]?[^"""]*?我[^"""]*?[询问问][：:]["""](.+?)["""]\s*$'
            match = re.search(i_action_ask_pattern, dialogue)
            if match:
                actual_dialogue = match.group(1).strip()
                dialogue = re.sub(i_action_ask_pattern, actual_dialogue, dialogue, count=1)
            
            # 模式11: "你"开头的动作描述+角色名+冒号+引号+实际对话
            # 例如: "你询问张师弟："这次试炼的策略你考虑得如何了？""
            you_action_role_pattern = r'["""]?你[^"""]*?[对向]?([^"""]+?)[：:]["""](.+?)["""]\s*$'
            match = re.search(you_action_role_pattern, dialogue)
            if match:
                actual_dialogue = match.group(2).strip()
                dialogue = re.sub(you_action_role_pattern, actual_dialogue, dialogue, count=1)
            
            # 模式12: "你"开头的动作描述+冒号+引号+实际对话
            # 例如: "你看着XX，轻声说："..."
            you_action_pattern = r'["""]?你[^"""]*?[说讲道询问问][：:]["""](.+?)["""]\s*$'
            match = re.search(you_action_pattern, dialogue)
            if match:
                actual_dialogue = match.group(1).strip()
                dialogue = re.sub(you_action_pattern, actual_dialogue, dialogue, count=1)
            
            # 清理多余的引号和空格
            dialogue = re.sub(r'^\s*["""]\s*', '', dialogue)
            dialogue = re.sub(r'\s*["""]\s*$', '', dialogue)
            dialogue = dialogue.strip()
        
        # 3. 如果描述只有角色名，使用默认描述
        if description and description.strip() in [self.role_name, self.role_id]:
            description = ""
        
        # 4. 确保描述不为空（如果为空，使用默认描述）
        if not description:
            if self.persona:
                if "稳重" in self.persona or "冷静" in self.persona:
                    description = "神情专注，语气沉稳"
                elif "活泼" in self.persona or "好奇" in self.persona:
                    description = "眼神闪烁，充满好奇"
                elif "坚定" in self.persona or "勇敢" in self.persona:
                    description = "目光坚定，充满决心"
                else:
                    description = "看向众人"
            else:
                description = "看向众人"
        
        # 最终清理：确保没有残留的格式标记
        dialogue = re.sub(r'\s*描述[：:]?\s*[（(].+?[）)]\s*', '', dialogue)
        dialogue = dialogue.strip()
        description = description.strip()
        
        return {
            "dialogue": dialogue,
            "description": description
        }
    
    def cleanup(self):
        """清理资源"""
        self.executor.shutdown(wait=False)

