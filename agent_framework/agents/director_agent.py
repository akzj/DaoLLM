"""
导演Agent - 负责发言顺序决策、玩家选择响应决策
"""
import json
from typing import List, Dict, Optional
import sys
from pathlib import Path
# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from agent_framework.agents.system_agent import SystemAgent


class DirectorAgent:
    """导演Agent - 剧情协调者"""
    
    def __init__(self, system_agent: SystemAgent):
        """
        初始化导演Agent
        
        Args:
            system_agent: 系统Agent实例
        """
        self.system_agent = system_agent
        self.current_scene_config: Optional[Dict] = None
        self.current_node_config: Optional[Dict] = None
    
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
        # 如果节点配置中有预设顺序，优先使用
        if self.current_node_config and "发言顺序" in self.current_node_config:
            speech_order = self.current_node_config["发言顺序"].copy()
            
            # 如果有玩家选择，插入玩家选项位置
            if player_choice and "玩家选项" in speech_order:
                # 找到"玩家选项"的位置，插入玩家选择后的角色回应
                player_response = self._decide_player_response(player_choice, current_state)
                if player_response:
                    idx = speech_order.index("玩家选项")
                    speech_order[idx] = "玩家选择"
                    # 在玩家选择后插入回应角色
                    speech_order.insert(idx + 1, player_response["trigger_role"])
                    print(f"[导演] 检测到玩家选择'{player_choice}'，触发角色'{player_response['trigger_role']}'回应")
                    return {
                        "speech_order": speech_order,
                        "player_response": player_response
                    }
            
            print(f"[导演] 使用预设发言顺序: {speech_order}")
            return {
                "speech_order": speech_order,
                "player_response": None
            }
        
        # 否则使用LLM决策
        print("[导演] 使用LLM决策发言顺序")
        return self._llm_decide_speech_order(current_state, player_choice)
    
    def _decide_player_response(self, player_choice: str, current_state: Dict) -> Optional[Dict]:
        """
        决策玩家选择的响应角色
        
        Args:
            player_choice: 玩家选择
            current_state: 当前状态
        
        Returns:
            {"trigger_role": "角色ID", "response_strategy": "策略"}
        """
        # 基于规则决策（简单策略）
        # 可以根据玩家选择的关键词匹配角色
        
        # 示例：如果玩家选择包含"不想"、"担忧"等，触发安抚角色
        if any(word in player_choice for word in ["不想", "担忧", "害怕", "担心"]):
            # 查找场景中的女性角色或友善角色
            if self.current_scene_config:
                roles = self.current_scene_config.get("角色列表", [])
                # 简单策略：选择第一个角色（实际应该基于角色人设）
                if roles:
                    return {
                        "trigger_role": roles[0],
                        "response_strategy": "安抚玩家，说明会保护他"
                    }
        
        return None
    
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
        
        # Qwen格式
        system_prompt = "你是一位修仙世界舞台剧的导演，负责协调角色发言顺序。"
        user_prompt = f"""当前剧情节点：{node_info}
当前角色列表：{', '.join(roles)}
当前状态：{json.dumps(current_state, ensure_ascii=False)}
玩家选择：{player_choice if player_choice else "无"}

请决策：
1. 下一个发言的角色是谁？（从角色列表中选择）
2. 发言顺序是什么？（列出后续3-5个角色的发言顺序）
3. 如果玩家选择了"{player_choice}"，哪个角色应该回应？回应策略是什么？

输出格式：JSON
{{
  "next_speaker": "角色名",
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
        
        # 解析JSON（简单实现，实际应该更robust）
        try:
            # 提取JSON部分
            json_start = generated.find("{")
            json_end = generated.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                result = json.loads(generated[json_start:json_end])
                return result
        except:
            pass
        
        # 如果解析失败，返回默认顺序
        return {
            "speech_order": roles[:3] if roles else [],
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

