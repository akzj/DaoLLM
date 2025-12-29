"""
导演Agent - 负责剧情发展决策、角色表演指导、发言顺序决策
"""
import json
import re
from typing import List, Dict, Optional
import sys
from pathlib import Path
# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from agent_framework.agents.system_agent import SystemAgent


def clean_json_string(json_str: str) -> str:
    """
    清理JSON字符串，移除尾随逗号等不合法字符
    
    Args:
        json_str: 原始JSON字符串
    
    Returns:
        清理后的JSON字符串
    """
    # 移除对象和数组中的尾随逗号
    # 匹配：,} 或 ,] 的情况
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    
    # 处理单引号：将字符串中的单引号转义或替换为双引号
    # 但要注意不要破坏已经正确转义的字符串
    # 简单策略：将未转义的单引号替换为双引号（在字符串值中）
    # 更安全的做法：使用正则表达式匹配字符串值，然后替换其中的单引号
    
    # 匹配字符串值中的单引号（不在引号内的单引号）
    # 这个比较复杂，先尝试简单的替换策略
    # 如果JSON解析失败，会在异常处理中处理
    
    return json_str


class DirectorAgent:
    """导演Agent - 剧情协调者，负责指导剧情发展和角色表演"""
    
    def __init__(self, system_agent: SystemAgent, use_llm_always: bool = False):
        """
        初始化导演Agent
        
        Args:
            system_agent: 系统Agent实例
            use_llm_always: 是否总是使用LLM决策（即使有预设顺序），默认False
        """
        self.system_agent = system_agent
        self.current_scene_config: Optional[Dict] = None
        self.current_node_config: Optional[Dict] = None
        self.use_llm_always = use_llm_always  # 新增：是否总是使用LLM
        self.recent_speeches: List[Dict] = []  # 记录最近的发言，用于上下文
    
    def set_scene(self, scene_config: Dict):
        """设置当前场景"""
        self.current_scene_config = scene_config
    
    def set_node(self, node_config: Dict):
        """设置当前节点"""
        self.current_node_config = node_config
    
    def decide_speech_order(
        self,
        current_state: Dict,
        player_choice: Optional[str] = None
    ) -> Dict:
        """
        决策发言顺序
        
        Args:
            current_state: 当前状态
            player_choice: 玩家选择（如有）
        
        Returns:
            {
                "speech_order": ["角色A", "角色B", ...],
                "player_response": {
                    "trigger_role": "角色C",
                    "response_strategy": "安抚玩家"
                } or None
            }
        """
        # 如果节点配置中有预设顺序
        if self.current_node_config and "发言顺序" in self.current_node_config:
            preset_order = self.current_node_config["发言顺序"].copy()
            
            # 如果设置了总是使用LLM，或者没有预设顺序，使用LLM决策
            if self.use_llm_always:
                print(f"[导演] 检测到预设顺序，但启用LLM模式，将基于预设顺序进行LLM决策")
                return self._llm_decide_speech_order_with_preset(current_state, player_choice, preset_order)
            
            # 如果有玩家选择，插入玩家选项位置
            if player_choice and "玩家选项" in preset_order:
                # 找到"玩家选项"的位置，插入玩家选择后的角色回应
                player_response = self._decide_player_response(player_choice, current_state)
                if player_response:
                    idx = preset_order.index("玩家选项")
                    preset_order[idx] = "玩家选择"
                    # 在玩家选择后插入回应角色
                    preset_order.insert(idx + 1, player_response["trigger_role"])
                    print(f"[导演] 检测到玩家选择'{player_choice}'，触发角色'{player_response['trigger_role']}'回应")
                    return {
                        "speech_order": preset_order,
                        "player_response": player_response
                    }
            
            print(f"[导演] 使用预设发言顺序: {preset_order}")
            return {
                "speech_order": preset_order,
                "player_response": None
            }
        
        # 否则使用LLM决策
        print("[导演] 使用LLM决策发言顺序（无预设顺序）")
        return self._llm_decide_speech_order(current_state, player_choice)
    
    def decide_player_response(self, player_choice: str, current_state: Dict) -> Optional[Dict]:
        """
        决策玩家选择的响应角色（使用LLM）
        
        Args:
            player_choice: 玩家选择
            current_state: 当前状态
        
        Returns:
            {"trigger_role": "角色ID", "response_strategy": "策略"}
        """
        # 构建Prompt
        node_info = json.dumps(self.current_node_config, ensure_ascii=False) if self.current_node_config else "无"
        roles = self.current_scene_config.get("角色列表", []) if self.current_scene_config else []
        
        system_prompt = """你是一位修仙世界舞台剧的导演，负责决定哪个角色应该回应玩家的选择。

你的职责：
1. 分析玩家的选择内容
2. 决定哪个角色最适合回应玩家的选择
3. 给出回应策略，指导角色如何回应"""
        
        user_prompt = f"""当前节点配置：{node_info}
当前角色列表：{', '.join(roles)}
当前状态：{json.dumps(current_state, ensure_ascii=False)}
玩家选择：{player_choice}

请分析玩家的选择，决定：
1. 哪个角色应该回应玩家的选择？（从角色列表中选择，或null表示无需回应）
2. 回应策略是什么？（如何回应，比如：安抚、鼓励、解释、同意等）

输出格式：JSON
{{
  "trigger_role": "角色名" or null,
  "response_strategy": "回应策略描述（如：安抚玩家，解释情况，鼓励参与等）"
}}"""
        
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # 使用系统Agent的模型生成
        with self.system_agent.lora_lock:
            inputs = self.system_agent.tokenizer(prompt, return_tensors="pt").to(self.system_agent.base_model.device)
            outputs = self.system_agent.base_model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
            )
        
        generated = self.system_agent.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        # 解析JSON
        try:
            json_start = generated.find("{")
            if json_start >= 0:
                brace_count = 0
                json_end = json_start
                for i in range(json_start, len(generated)):
                    if generated[i] == '{':
                        brace_count += 1
                    elif generated[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
                
                if json_end > json_start:
                    json_str = generated[json_start:json_end]
                    # 清理JSON字符串（移除尾随逗号等）
                    json_str = clean_json_string(json_str)
                    
                    # 修复单引号问题
                    json_str = re.sub(r"'([^']*)'", r'"\1"', json_str)
                    
                    try:
                        result = json.loads(json_str)
                    except json.JSONDecodeError:
                        json_str = json_str.replace('\n', '\\n').replace('\r', '\\r')
                        result = json.loads(json_str)
                    
                    # 验证结果
                    if "trigger_role" in result:
                        trigger_role = result.get("trigger_role")
                        # 如果trigger_role不为null，验证是否是有效角色
                        if trigger_role and trigger_role not in roles:
                            print(f"[导演] 警告：无效的角色 '{trigger_role}'，清除回应")
                            return None
                        
                        if trigger_role:
                            return {
                                "trigger_role": trigger_role,
                                "response_strategy": result.get("response_strategy", "回应玩家的选择")
                            }
                        else:
                            return None  # 无需回应
        except json.JSONDecodeError as e:
            print(f"[导演] 玩家回应决策JSON解析失败: {e}")
            print(f"[导演] 原始输出片段: {generated[:200]}...")
        except Exception as e:
            print(f"[导演] 玩家回应决策失败: {e}")
        
        # 如果解析失败，使用简单规则
        # 如果玩家选择包含"不想"、"担忧"等，触发第一个角色回应
        if any(word in player_choice for word in ["不想", "担忧", "害怕", "担心", "不安"]):
            if roles:
                return {
                    "trigger_role": roles[0],
                    "response_strategy": "安抚玩家，鼓励参与"
                }
        
        return None
    
    def _decide_player_response(self, player_choice: str, current_state: Dict) -> Optional[Dict]:
        """
        决策玩家选择的响应角色（旧方法，保留兼容性，调用新方法）
        
        Args:
            player_choice: 玩家选择
            current_state: 当前状态
        
        Returns:
            {"trigger_role": "角色ID", "response_strategy": "策略"}
        """
        # 调用新的decide_player_response方法
        return self.decide_player_response(player_choice, current_state)
    
    def _llm_decide_speech_order_with_preset(
        self,
        current_state: Dict,
        player_choice: Optional[str],
        preset_order: List[str]
    ) -> Dict:
        """
        基于预设顺序使用LLM进行决策（可以调整或完全重新决策）
        
        Args:
            current_state: 当前状态
            player_choice: 玩家选择
            preset_order: 预设的发言顺序
        """
        # 构建Prompt，包含预设顺序作为参考
        node_info = json.dumps(self.current_node_config, ensure_ascii=False) if self.current_node_config else "无"
        roles = self.current_scene_config.get("角色列表", []) if self.current_scene_config else []
        
        system_prompt = "你是一位修仙世界舞台剧的导演，负责协调角色发言顺序。你可以基于预设顺序进行调整，或完全重新决策。"
        user_prompt = f"""当前剧情节点：{node_info}
当前角色列表：{', '.join(roles)}
当前状态：{json.dumps(current_state, ensure_ascii=False)}
玩家选择：{player_choice if player_choice else "无"}
预设发言顺序（仅供参考）：{preset_order}

请决策：
1. 基于当前剧情状态和玩家选择，决定最合适的发言顺序
2. 可以完全遵循预设顺序，也可以根据剧情需要调整
3. 如果玩家选择了"{player_choice}"，哪个角色应该回应？回应策略是什么？

输出格式：JSON
{{
  "speech_order": ["角色1", "角色2", "玩家选项", "角色3"],
  "player_response": {{
    "trigger_role": "角色名",
    "response_strategy": "策略描述"
  }} or null
}}"""
        
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # 使用系统Agent的模型生成
        with self.system_agent.lora_lock:
            inputs = self.system_agent.tokenizer(prompt, return_tensors="pt").to(self.system_agent.base_model.device)
            outputs = self.system_agent.base_model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=0.7,
                do_sample=True,
            )
        
        generated = self.system_agent.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        # 解析JSON - 改进的解析逻辑
        try:
            # 方法1：尝试找到第一个完整的JSON对象
            json_start = generated.find("{")
            if json_start >= 0:
                # 找到匹配的右括号
                brace_count = 0
                json_end = json_start
                for i in range(json_start, len(generated)):
                    if generated[i] == '{':
                        brace_count += 1
                    elif generated[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
                
                if json_end > json_start:
                    json_str = generated[json_start:json_end]
                    # 清理JSON字符串（移除尾随逗号等）
                    json_str = clean_json_string(json_str)
                    
                    # 修复单引号问题
                    json_str = re.sub(r"'([^']*)'", r'"\1"', json_str)
                    
                    try:
                        result = json.loads(json_str)
                    except json.JSONDecodeError:
                        json_str = json_str.replace('\n', '\\n').replace('\r', '\\r')
                        result = json.loads(json_str)
                    
                    # 验证和清理结果
                    if "speech_order" in result:
                        speech_order = result["speech_order"]
                        # 验证发言顺序：过滤无效角色，确保只包含有效角色或"玩家选项"
                        valid_roles = self.current_scene_config.get("角色列表", []) if self.current_scene_config else []
                        valid_special = ["玩家选项", "玩家选择"]
                        
                        # 检查节点是否需要玩家选项
                        has_player_options = False
                        if self.current_node_config and "玩家选项" in self.current_node_config:
                            has_player_options = True
                        
                        cleaned_order = []
                        has_player_option_marker = False
                        
                        for item in speech_order:
                            item_str = str(item).strip()
                            
                            # 检查是否是有效角色
                            if item_str in valid_roles:
                                cleaned_order.append(item_str)
                            # 检查是否是特殊标记
                            elif item_str in valid_special:
                                if item_str == "玩家选项":
                                    has_player_option_marker = True
                                cleaned_order.append(item_str)
                            else:
                                # 如果包含"选项"或"选择"关键词，可能是玩家选项的文本描述
                                if "选项" in item_str or "选择" in item_str or "询问" in item_str or "同意" in item_str or "建议" in item_str:
                                    # 如果节点需要玩家选项但还没有标记，添加"玩家选项"
                                    if has_player_options and not has_player_option_marker:
                                        cleaned_order.append("玩家选项")
                                        has_player_option_marker = True
                                        print(f"[导演] 将 '{item_str}' 识别为玩家选项位置，已替换为'玩家选项'")
                                    else:
                                        print(f"[导演] 警告：过滤掉无效项 '{item_str}'（可能是玩家选项文本）")
                                else:
                                    print(f"[导演] 警告：过滤掉无效角色 '{item_str}'")
                        
                        # 如果节点需要玩家选项但没有标记，添加它
                        if has_player_options and not has_player_option_marker:
                            # 在适当位置插入（通常在中间或后半部分）
                            insert_pos = min(len(cleaned_order), max(1, len(cleaned_order) // 2))
                            cleaned_order.insert(insert_pos, "玩家选项")
                            print(f"[导演] 自动添加'玩家选项'标记到位置 {insert_pos}")
                        
                        # 如果清理后为空，使用预设顺序
                        if not cleaned_order:
                            print(f"[导演] LLM返回的发言顺序无效，回退到预设顺序")
                            cleaned_order = preset_order
                        
                        result["speech_order"] = cleaned_order
                        print(f"[导演] LLM决策的发言顺序: {cleaned_order}")
                        return result
        except json.JSONDecodeError as e:
            print(f"[导演] LLM决策JSON解析失败: {e}")
            print(f"[导演] 原始输出片段: {generated[:200]}...")
        except Exception as e:
            print(f"[导演] LLM决策解析失败: {e}")
        
        # 如果解析失败，返回预设顺序
        print(f"[导演] 回退到预设顺序: {preset_order}")
        return {
            "speech_order": preset_order,
            "player_response": None
        }
    
    def _llm_decide_speech_order(
        self,
        current_state: Dict,
        player_choice: Optional[str] = None
    ) -> Dict:
        """
        使用LLM决策发言顺序
        
        Args:
            current_state: 当前状态
            player_choice: 玩家选择
        """
        # 构建Prompt
        node_info = json.dumps(self.current_node_config, ensure_ascii=False) if self.current_node_config else "无"
        roles = self.current_scene_config.get("角色列表", []) if self.current_scene_config else []
        
        # 检查节点是否有玩家选项
        has_player_options = False
        player_options_list = []
        if self.current_node_config and "玩家选项" in self.current_node_config:
            has_player_options = True
            player_options_list = self.current_node_config.get("玩家选项", [])
        
        # Qwen格式
        system_prompt = "你是一位修仙世界舞台剧的导演，负责协调角色发言顺序。你必须严格遵守输出格式，只返回有效的角色名或特殊标记。"
        
        player_options_hint = ""
        if has_player_options:
            player_options_hint = f"\n重要：此节点需要玩家选择，玩家选项有：{player_options_list}\n你必须在speech_order中包含一次且仅一次\"玩家选项\"标记。"
        
        user_prompt = f"""当前剧情节点：{node_info}
当前角色列表：{', '.join(roles)}
当前状态：{json.dumps(current_state, ensure_ascii=False)}
玩家选择：{player_choice if player_choice else "无"}
{player_options_hint}

重要规则：
1. speech_order 中的每个元素必须是以下之一：
   - 有效的角色名（从角色列表中）：{', '.join(roles)}
   - "玩家选项"（如果节点需要玩家选择，必须包含一次且仅一次）
2. 不能包含其他任何文本，如"询问秘境情况"、"同意"等选项文本
3. 如果节点需要玩家选择，必须在适当位置插入"玩家选项"
4. 发言顺序应该符合剧情节奏，通常3-6个元素

请决策：
1. 基于当前剧情状态，决定最合适的发言顺序
2. 如果节点需要玩家选择，在适当位置插入"玩家选项"
3. 如果玩家选择了"{player_choice}"，哪个角色应该回应？回应策略是什么？

输出格式：JSON（必须严格遵守）
{{
  "speech_order": ["角色1", "角色2", "玩家选项", "角色3"],
  "player_response": {{
    "trigger_role": "角色名",
    "response_strategy": "策略描述"
  }} or null
}}"""
        
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # 使用系统Agent的模型生成
        with self.system_agent.lora_lock:
            inputs = self.system_agent.tokenizer(prompt, return_tensors="pt").to(self.system_agent.base_model.device)
            outputs = self.system_agent.base_model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=0.7,
                do_sample=True,
            )
        
        generated = self.system_agent.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        # 解析JSON - 改进的解析逻辑
        try:
            # 方法1：尝试找到第一个完整的JSON对象
            json_start = generated.find("{")
            if json_start >= 0:
                # 找到匹配的右括号
                brace_count = 0
                json_end = json_start
                for i in range(json_start, len(generated)):
                    if generated[i] == '{':
                        brace_count += 1
                    elif generated[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
                
                if json_end > json_start:
                    json_str = generated[json_start:json_end]
                    # 清理JSON字符串（移除尾随逗号等）
                    json_str = clean_json_string(json_str)
                    
                    # 修复单引号问题
                    json_str = re.sub(r"'([^']*)'", r'"\1"', json_str)
                    
                    try:
                        result = json.loads(json_str)
                    except json.JSONDecodeError:
                        json_str = json_str.replace('\n', '\\n').replace('\r', '\\r')
                        result = json.loads(json_str)
                    
                    # 验证结果格式
                    if "speech_order" in result:
                        speech_order = result["speech_order"]
                        # 验证和清理发言顺序
                        valid_roles = self.current_scene_config.get("角色列表", []) if self.current_scene_config else []
                        valid_special = ["玩家选项", "玩家选择"]
                        
                        # 检查节点是否需要玩家选项
                        has_player_options = False
                        if self.current_node_config and "玩家选项" in self.current_node_config:
                            has_player_options = True
                        
                        cleaned_order = []
                        has_player_option_marker = False
                        
                        for item in speech_order:
                            item_str = str(item).strip()
                            
                            # 检查是否是有效角色
                            if item_str in valid_roles:
                                cleaned_order.append(item_str)
                            # 检查是否是特殊标记
                            elif item_str in valid_special:
                                if item_str == "玩家选项":
                                    has_player_option_marker = True
                                cleaned_order.append(item_str)
                            else:
                                # 如果包含"选项"、"选择"、"询问"、"同意"、"建议"等关键词，可能是玩家选项的文本描述
                                if any(keyword in item_str for keyword in ["选项", "选择", "询问", "同意", "建议", "表示", "保持", "观察", "警惕"]):
                                    # 如果节点需要玩家选项但还没有标记，添加"玩家选项"
                                    if has_player_options and not has_player_option_marker:
                                        cleaned_order.append("玩家选项")
                                        has_player_option_marker = True
                                        print(f"[导演] 将 '{item_str}' 识别为玩家选项位置，已替换为'玩家选项'")
                                    else:
                                        print(f"[导演] 警告：过滤掉无效项 '{item_str}'（可能是玩家选项文本）")
                                else:
                                    print(f"[导演] 警告：过滤掉无效角色 '{item_str}'")
                        
                        # 如果节点需要玩家选项但没有标记，自动添加
                        if has_player_options and not has_player_option_marker:
                            # 在适当位置插入（通常在中间或后半部分）
                            insert_pos = min(len(cleaned_order), max(1, len(cleaned_order) // 2))
                            cleaned_order.insert(insert_pos, "玩家选项")
                            print(f"[导演] 自动添加'玩家选项'标记到位置 {insert_pos}")
                        
                        # 如果清理后为空，使用默认顺序
                        if not cleaned_order:
                            cleaned_order = roles[:3] if roles else []
                            print(f"[导演] LLM返回的发言顺序无效，使用默认顺序")
                        
                        result["speech_order"] = cleaned_order
                        print(f"[导演] LLM决策的发言顺序: {cleaned_order}")
                        return result
        except json.JSONDecodeError as e:
            print(f"[导演] LLM决策JSON解析失败: {e}")
            print(f"[导演] 原始输出片段: {generated[:200]}...")
        except Exception as e:
            print(f"[导演] LLM决策解析失败: {e}")
        
        # 如果解析失败，返回默认顺序
        default_order = roles[:3] if roles else []
        print(f"[导演] 解析失败，使用默认顺序: {default_order}")
        return {
            "speech_order": default_order,
            "player_response": None
        }
    
    def check_scene_end_condition(self) -> bool:
        """检查场景结束条件"""
        if not self.current_node_config:
            return False
        
        # 检查节点是否有下一节点
        next_node = self.current_node_config.get("下一节点")
        return next_node is None
    
    def get_next_scene(self) -> Optional[str]:
        """获取下一场景"""
        if self.current_scene_config:
            return self.current_scene_config.get("下一场景")
        return None
    
    def generate_player_options(
        self,
        current_state: Dict,
        context: str = ""
    ) -> List[str]:
        """
        生成玩家选项（由导演Agent根据当前剧情动态生成）
        
        Args:
            current_state: 当前状态
            context: 当前上下文
        
        Returns:
            玩家选项列表
        """
        # 构建Prompt
        node_info = json.dumps(self.current_node_config, ensure_ascii=False) if self.current_node_config else "无"
        scene_info = json.dumps(self.current_scene_config, ensure_ascii=False) if self.current_scene_config else "无"
        
        system_prompt = """你是一位修仙世界舞台剧的导演，负责为玩家生成合适的选项。

你的职责：
1. 根据当前剧情节点和剧情发展，为玩家生成3-5个选项
2. 选项应该符合当前剧情，有明确的行动意图
3. 选项应该简洁明了（10-20字），易于理解
4. 选项应该能够推动剧情发展或展现玩家个性"""
        
        user_prompt = f"""当前场景配置：{scene_info}
当前节点配置：{node_info}
当前状态：{json.dumps(current_state, ensure_ascii=False)}
当前上下文：{context[:500]}...

请为玩家生成3-5个选项，要求：
1. 符合当前剧情节点和目标
2. 每个选项10-20字，简洁明了
3. 有明确的行动意图
4. 能够推动剧情发展或展现玩家个性

输出格式：JSON
{{
  "options": ["选项1", "选项2", "选项3", "选项4"]
}}"""
        
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # 使用系统Agent的模型生成
        with self.system_agent.lora_lock:
            inputs = self.system_agent.tokenizer(prompt, return_tensors="pt").to(self.system_agent.base_model.device)
            outputs = self.system_agent.base_model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.8,
                do_sample=True,
            )
        
        generated = self.system_agent.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        # 解析JSON
        try:
            json_start = generated.find("{")
            if json_start >= 0:
                brace_count = 0
                json_end = json_start
                for i in range(json_start, len(generated)):
                    if generated[i] == '{':
                        brace_count += 1
                    elif generated[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
                
                if json_end > json_start:
                    json_str = generated[json_start:json_end]
                    # 清理JSON字符串（移除尾随逗号等）
                    json_str = clean_json_string(json_str)
                    
                    # 修复单引号问题
                    json_str = re.sub(r"'([^']*)'", r'"\1"', json_str)
                    
                    try:
                        result = json.loads(json_str)
                    except json.JSONDecodeError:
                        json_str = json_str.replace('\n', '\\n').replace('\r', '\\r')
                        result = json.loads(json_str)
                    
                    if "options" in result and isinstance(result["options"], list):
                        options = result["options"]
                        # 验证和清理选项
                        cleaned_options = []
                        for opt in options:
                            opt_str = str(opt).strip()
                            if opt_str and len(opt_str) > 2:  # 至少3个字符
                                cleaned_options.append(opt_str)
                        
                        if cleaned_options:
                            print(f"[导演] 生成玩家选项: {cleaned_options}")
                            return cleaned_options[:5]  # 最多5个选项
        except json.JSONDecodeError as e:
            print(f"[导演] 玩家选项JSON解析失败: {e}")
            print(f"[导演] 原始输出片段: {generated[:200]}...")
        except Exception as e:
            print(f"[导演] 玩家选项生成失败: {e}")
        
        # 如果解析失败，使用默认选项
        default_options = ["继续观察", "保持沉默", "询问情况", "表达想法"]
        print(f"[导演] 使用默认玩家选项: {default_options}")
        return default_options
    
    def add_speech_record(self, role_id: str, speech_content: str, speech_type: str = "对话"):
        """记录角色发言，用于导演决策"""
        self.recent_speeches.append({
            "role": role_id,
            "content": speech_content,
            "type": speech_type
        })
        # 只保留最近5条发言
        if len(self.recent_speeches) > 5:
            self.recent_speeches.pop(0)
    
    def decide_next_action(
        self,
        current_state: Dict,
        last_speaker: Optional[str] = None,
        last_speech: Optional[str] = None,
        just_handled_player_choice: bool = False
    ) -> Dict:
        """
        决定下一步剧情发展（每次发言后调用）
        
        Args:
            current_state: 当前状态
            last_speaker: 上一个发言的角色
            last_speech: 上一个发言的内容
            just_handled_player_choice: 是否刚刚处理完玩家选择
        
        Returns:
            {
                "action_type": "继续发言" | "切换节点" | "结束场景" | "玩家选项",
                "next_speaker": "角色名" or None,
                "acting_guidance": {
                    "role": "角色名",
                    "guidance": "表演指导",
                    "emotion": "情绪",
                    "focus": "关注点"
                } or None,
                "plot_advancement": "剧情推进建议",
                "script_alignment": "剧本对齐检查"
            }
        """
        # 构建Prompt
        node_info = json.dumps(self.current_node_config, ensure_ascii=False) if self.current_node_config else "无"
        scene_info = json.dumps(self.current_scene_config, ensure_ascii=False) if self.current_scene_config else "无"
        roles = self.current_scene_config.get("角色列表", []) if self.current_scene_config else []
        
        # 获取最近的发言历史
        recent_history = "\n".join([
            f"- [{s['role']}]: {s['content'][:100]}..." 
            for s in self.recent_speeches[-3:]
        ]) if self.recent_speeches else "无"
        
        # 获取记忆上下文
        memory_context = self.system_agent.memory_system.get_context_text(
            scene=self.system_agent.current_scene
        )[:500]  # 限制长度
        
        system_prompt = """你是一位修仙世界舞台剧的导演，负责指导剧情发展和角色表演。

你的职责：
1. **剧情发展决策**：每次角色发言后，决定下一步剧情如何发展
2. **角色表演指导**：指导角色如何演戏，包括情绪、动作、对话风格
3. **剧本对齐**：确保剧情不偏离剧本要求
4. **节奏控制**：控制剧情节奏，适时推进或停留

重要原则：
- 必须严格遵循剧本要求，不能偏离
- 每次发言后都要决定下一步
- 指导角色表演要具体、生动
- 保持剧情连贯性和逻辑性"""
        
        # 计算发言次数，用于决定是否需要玩家选项
        speech_count = len(self.recent_speeches)
        need_player_option = False
        if speech_count > 0:
            # 每3-5次发言后，考虑插入玩家选项
            if speech_count % 4 == 0 or (speech_count >= 3 and speech_count % 3 == 0):
                need_player_option = True
        
        player_option_hint = ""
        if just_handled_player_choice:
            player_option_hint = "\n⚠️ **重要提示**：刚刚处理完玩家选择，角色已回应。现在应该继续角色对话，让其他角色也参与讨论，不要立即再次显示玩家选项。至少让2-3个角色发言后再考虑玩家选项。"
        elif need_player_option:
            player_option_hint = "\n⚠️ **重要提示**：当前已有多轮角色对话，建议插入玩家选项让玩家参与剧情。选择\"玩家选项\"可以让玩家做出选择，增加互动性。"
        
        user_prompt = f"""当前场景配置：{scene_info}
当前节点配置：{node_info}
当前角色列表：{', '.join(roles)}
当前状态：{json.dumps(current_state, ensure_ascii=False)}
最近发言历史：
{recent_history}
记忆上下文：{memory_context[:300]}...
上一个发言者：{last_speaker if last_speaker else "无"}
上一个发言内容：{last_speech[:200] if last_speech else "无"}...
当前发言轮数：{speech_count}
{player_option_hint}

请作为导演，分析当前情况并决定下一步：

1. **剧情发展决策**：
   - 如果节点目标未达成，继续推进（选择"继续发言"）
   - 如果节点目标已达成，切换到下一节点（选择"切换节点"）
   - 如果场景已完成，结束场景（选择"结束场景"）
   - **如果剧情需要玩家参与或已有多轮对话，插入玩家选项（选择"玩家选项"）** ⭐ 重要：每3-5轮对话后应该让玩家参与
   - ⚠️ **如果刚刚处理完玩家选择（just_handled_player_choice=True），必须选择"继续发言"，让其他角色也参与讨论，不要立即再次显示玩家选项**

2. **角色表演指导**（如果选择"继续发言"）：
   - 指定下一个发言的角色
   - 给出具体的表演指导：情绪、动作、对话风格
   - 确保符合角色人设和当前剧情

3. **剧本对齐检查**：
   - 检查当前剧情是否偏离剧本
   - 如果偏离，给出纠正建议

输出格式：JSON（必须严格遵守）
{{
  "action_type": "继续发言" | "切换节点" | "结束场景" | "玩家选项",
  "next_speaker": "角色名" or null,
  "acting_guidance": {{
    "role": "角色名",
    "guidance": "具体的表演指导，包括情绪、动作、对话风格",
    "emotion": "情绪状态（如：紧张、兴奋、担忧等）",
    "focus": "关注点（如：观察石碑、关心队友等）"
  }} or null,
  "plot_advancement": "剧情推进建议（1-2句话）",
  "script_alignment": "剧本对齐检查（是否偏离，如何纠正）",
  "reasoning": "决策理由（为什么这样决定）"
}}"""
        
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # 使用系统Agent的模型生成
        with self.system_agent.lora_lock:
            inputs = self.system_agent.tokenizer(prompt, return_tensors="pt").to(self.system_agent.base_model.device)
            outputs = self.system_agent.base_model.generate(
                **inputs,
                max_new_tokens=600,
                temperature=0.7,
                do_sample=True,
            )
        
        generated = self.system_agent.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        # 解析JSON，带重试机制
        max_retries = 3
        for retry in range(max_retries):
            try:
                json_start = generated.find("{")
                if json_start >= 0:
                    brace_count = 0
                    json_end = json_start
                    for i in range(json_start, len(generated)):
                        if generated[i] == '{':
                            brace_count += 1
                        elif generated[i] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_end = i + 1
                                break
                    
                    if json_end > json_start:
                        json_str = generated[json_start:json_end]
                        
                        # 清理JSON字符串：移除开头和结尾的引号（如果被单引号或双引号包围）
                        json_str = json_str.strip()
                        
                        # 检查并移除开头和结尾的引号
                        if json_str.startswith("'") and json_str.endswith("'"):
                            json_str = json_str[1:-1].strip()
                        elif json_str.startswith('"') and json_str.endswith('"'):
                            json_str = json_str[1:-1].strip()
                        
                        # 确保JSON字符串以 { 开头
                        if not json_str.startswith('{'):
                            # 尝试找到第一个 {
                            first_brace = json_str.find('{')
                            if first_brace >= 0:
                                json_str = json_str[first_brace:]
                            else:
                                # 如果找不到 {，尝试从原始字符串重新提取
                                raise ValueError("JSON字符串不以 { 开头")
                        
                        # 清理JSON字符串（移除尾随逗号等）
                        json_str = clean_json_string(json_str)
                        
                        # 修复单引号问题：将字符串值中的单引号替换为转义的双引号
                        json_str = re.sub(r"'([^']*)'", r'"\1"', json_str)
                        
                        # 如果还有问题，尝试更复杂的修复
                        try:
                            result = json.loads(json_str)
                        except json.JSONDecodeError as e:
                            # 如果还是失败，尝试修复常见的JSON错误
                            original_error = str(e)
                            
                            # 1. 移除字符串值中的换行符（保留转义的\n）
                            json_str_fixed = json_str.replace('\n', '\\n').replace('\r', '\\r')
                            # 2. 修复可能的尾随逗号（在对象和数组中）
                            json_str_fixed = re.sub(r',\s*}', '}', json_str_fixed)
                            json_str_fixed = re.sub(r',\s*]', ']', json_str_fixed)
                            # 3. 移除可能的控制字符
                            json_str_fixed = ''.join(char for char in json_str_fixed if ord(char) >= 32 or char in '\n\r\t')
                            
                            # 4. 再次尝试解析
                            try:
                                result = json.loads(json_str_fixed)
                            except json.JSONDecodeError:
                                # 5. 如果还是失败，尝试提取第一个完整的JSON对象
                                match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', json_str_fixed)
                                if match:
                                    json_str_fixed = match.group(0)
                                    try:
                                        result = json.loads(json_str_fixed)
                                    except json.JSONDecodeError:
                                        # 最后尝试：移除所有可能的非法字符
                                        json_str_fixed = re.sub(r'[^\x20-\x7E\n\r\t]', '', json_str_fixed)
                                        result = json.loads(json_str_fixed)
                                else:
                                    raise e
                        
                        # 验证结果
                        if "action_type" in result:
                            # 验证action_type
                            valid_actions = ["继续发言", "切换节点", "结束场景", "玩家选项"]
                            if result["action_type"] not in valid_actions:
                                print(f"[导演] 警告：无效的action_type '{result['action_type']}'，使用默认值")
                                result["action_type"] = "继续发言"
                            
                            # 验证next_speaker
                            if result.get("next_speaker"):
                                if result["next_speaker"] not in roles:
                                    print(f"[导演] 警告：无效的角色 '{result['next_speaker']}'，清除")
                                    result["next_speaker"] = None
                            
                            # 打印决策信息
                            print(f"\n[导演] 剧情发展决策：{result.get('action_type', '未知')}")
                            if result.get("plot_advancement"):
                                print(f"[导演] 剧情推进：{result['plot_advancement']}")
                            if result.get("script_alignment"):
                                print(f"[导演] 剧本对齐：{result['script_alignment']}")
                            if result.get("acting_guidance"):
                                guidance = result["acting_guidance"]
                                print(f"[导演] 表演指导 - {guidance.get('role', '未知')}：{guidance.get('guidance', '无')}")
                            
                            return result
            except (json.JSONDecodeError, ValueError) as e:
                if retry < max_retries - 1:
                    print(f"[导演] JSON解析失败（尝试 {retry + 1}/{max_retries}），重试生成...")
                    # 重新生成
                    with self.system_agent.lora_lock:
                        inputs = self.system_agent.tokenizer(prompt, return_tensors="pt").to(self.system_agent.base_model.device)
                        outputs = self.system_agent.base_model.generate(
                            **inputs,
                            max_new_tokens=600,
                            temperature=0.7,
                            do_sample=True,
                        )
                    generated = self.system_agent.tokenizer.decode(
                        outputs[0][inputs.input_ids.shape[1]:],
                        skip_special_tokens=True
                    )
                    continue
                else:
                    print(f"[导演] 决策JSON解析失败（已重试{max_retries}次）: {e}")
                    print(f"[导演] 原始输出片段: {generated[:200]}...")
            except Exception as e:
                print(f"[导演] 决策解析失败: {e}")
                break
        
        # 如果所有重试都失败，返回默认决策：继续发言
        default_speaker = roles[0] if roles else None
        print(f"[导演] 解析失败，使用默认决策：继续发言，下一个发言者：{default_speaker}")
        return {
            "action_type": "继续发言",
            "next_speaker": default_speaker,
            "acting_guidance": None,
            "plot_advancement": "继续推进当前节点剧情",
            "script_alignment": "保持当前节奏",
            "reasoning": "LLM解析失败，使用默认决策"
        }

