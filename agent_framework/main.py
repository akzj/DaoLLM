"""
主流程控制 - 剧情演绎引擎
"""
import asyncio
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agent_framework.agents.system_agent import SystemAgent
from agent_framework.agents.director_agent import DirectorAgent
from agent_framework.agents.role_agent import RoleAgent


class DramaEngine:
    """剧情演绎引擎"""
    
    def __init__(
        self,
        scenario_path: str,
        characters_path: str,
        base_model_name: str = "Qwen/Qwen1.5-7B-Chat"
    ):
        """
        初始化剧情引擎
        
        Args:
            scenario_path: 剧本配置文件路径
            characters_path: 角色配置文件路径
            base_model_name: 基础模型名称
        """
        # 加载配置
        self.scenario_config = self._load_yaml(scenario_path)
        self.characters_config = self._load_yaml(characters_path)
        
        # 验证配置
        from agent_framework.utils.config_validator import validate_configs
        validate_configs(self.scenario_config, self.characters_config)
        
        # 初始化Agent
        self.system_agent = SystemAgent(base_model_name=base_model_name, use_4bit=True)
        # 启用LLM模式：即使有预设顺序，导演Agent也会使用LLM进行决策和调整
        self.director_agent = DirectorAgent(self.system_agent, use_llm_always=True)
        
        # 初始化角色Agent
        self.role_agents: Dict[str, RoleAgent] = {}
        for role_config in self.characters_config["角色"]:
            role_id = role_config["角色ID"]
            self.role_agents[role_id] = RoleAgent(self.system_agent, role_config)
        
        # 当前状态
        self.current_scene_idx = 0
        self.current_node_idx = 0
        self.player_choice: Optional[str] = None
    
    def _load_yaml(self, path: str) -> Dict:
        """加载YAML配置"""
        script_dir = Path(__file__).parent
        full_path = script_dir / path if not Path(path).is_absolute() else Path(path)
        
        with open(full_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    async def start(self):
        """启动剧情引擎"""
        print("=" * 60)
        print("修仙世界剧情演绎开始")
        print("=" * 60)
        
        # 从第一个场景开始
        await self._enter_scene(0)
    
    async def _enter_scene(self, scene_idx: int):
        """进入场景"""
        scenes = self.scenario_config["剧本"]["场景列表"]
        if scene_idx >= len(scenes):
            print("\n剧情结束")
            return
        
        scene_config = scenes[scene_idx]
        scene_id = scene_config.get("场景ID", f"场景{scene_idx+1}")
        
        # 设置当前场景
        self.system_agent.current_scene = scene_id
        self.director_agent.set_scene(scene_config)
        self.current_scene_idx = scene_idx
        
        # 显示场景描述
        scene_desc = self.system_agent.generate_scene_description(scene_config)
        print(f"\n[场景切换] 进入场景: {scene_id}\n")
        print(f"{scene_desc}\n")
        
        # 加载场景角色LoRA
        role_list = scene_config.get("角色列表", [])
        for role_id in role_list:
            if role_id in self.role_agents:
                role_config = next(
                    (r for r in self.characters_config["角色"] if r["角色ID"] == role_id),
                    None
                )
                if role_config:
                    lora_path = role_config.get("LoRA路径", "")
                    if lora_path:
                        self.system_agent.load_lora(role_id, lora_path, verbose=False)
        
        # 进入第一个节点
        await self._enter_node(scene_config, 0)
    
    async def _enter_node(self, scene_config: Dict, node_idx: int):
        """进入节点"""
        nodes = scene_config.get("节点列表", [])
        if node_idx >= len(nodes):
            # 场景结束，检查是否需要切换场景
            await self._check_scene_end(scene_config)
            return
        
        node_config = nodes[node_idx]
        node_id = node_config.get("节点ID", f"节点{node_idx+1}")
        
        print(f"\n[节点] {node_id}\n")
        
        # 设置当前节点
        self.director_agent.set_node(node_config)
        self.current_node_idx = node_idx
        
        # 执行节点
        await self._execute_node(node_config, scene_config)
    
    async def _execute_node(self, node_config: Dict, scene_config: Dict):
        """执行节点 - 由导演在每次发言后决定下一个发言者"""
        current_state = {
            "scene": self.system_agent.current_scene,
            "node": node_config.get("节点ID"),
            "global_state": self.system_agent.global_state
        }
        
        roles = scene_config.get("角色列表", [])
        if not roles:
            print("[警告] 场景中没有角色，跳过节点")
            return
        
        # 节点开始时，由导演决定第一个发言者
        director_decision = self.director_agent.decide_next_action(
            current_state=current_state,
            last_speaker=None,
            last_speech=None
        )
        
        # 获取第一个发言者
        next_speaker = director_decision.get("next_speaker")
        if not next_speaker or next_speaker not in roles:
            # 如果没有指定，使用第一个角色
            next_speaker = roles[0]
        
        # 动态执行：每次发言后由导演决定下一个发言者
        max_iterations = 50  # 防止无限循环
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # 如果当前发言者是"玩家选项"，处理玩家选择
            if next_speaker == "玩家选项":
                # 由导演Agent生成玩家选项
                context = self.system_agent.memory_system.get_context_text(
                    scene=self.system_agent.current_scene
                )
                options = self.director_agent.generate_player_options(current_state, context)
                
                # 异步获取玩家选择（函数内部会显示选项）
                from agent_framework.utils.player_input import get_player_choice_async
                self.player_choice = await get_player_choice_async(options)
                
                if self.player_choice:
                    print(f"\n[玩家选择] {self.player_choice}\n")
                else:
                    # 如果玩家取消或超时，使用默认选项
                    self.player_choice = options[0] if options else "继续观察"
                    print(f"\n[使用默认选项] {self.player_choice}\n")
                
                # 保存玩家选择到记忆
                self.system_agent.memory_system.add_memory(
                    content=f"玩家选择：{self.player_choice}",
                    memory_type="玩家选择",
                    scene=self.system_agent.current_scene,
                    importance=0.6
                )
                
                # 玩家选择后，由导演Agent决定哪个角色应该回应
                player_response = self.director_agent.decide_player_response(
                    self.player_choice,
                    current_state
                )
                
                # 如果有玩家响应角色，生成响应
                if player_response:
                    trigger_role = player_response["trigger_role"]
                    response_strategy = player_response.get("response_strategy", "回应玩家的选择")
                    print(f"\n[导演] 玩家选择触发角色'{trigger_role}'回应，策略：{response_strategy}\n")
                    if trigger_role in self.role_agents:
                        speech_result = await self._generate_role_speech(
                            trigger_role,
                            node_config,
                            response_strategy
                        )
                        # 回应后，由导演决定下一步
                        speech_text = ""
                        if speech_result:
                            speech_text = speech_result.get("dialogue", "") + " " + speech_result.get("description", "")
                        self.director_agent.add_speech_record(trigger_role, speech_text)
                        
                        # 更新状态
                        current_state = {
                            "scene": self.system_agent.current_scene,
                            "node": node_config.get("节点ID"),
                            "global_state": self.system_agent.global_state
                        }
                        
                        # 玩家选择后，角色回应后，应该继续角色对话，而不是立即再次显示玩家选项
                        # 至少让其他角色也发言一轮
                        director_decision = self.director_agent.decide_next_action(
                            current_state=current_state,
                            last_speaker=trigger_role,
                            last_speech=speech_text,
                            just_handled_player_choice=True  # 标记刚刚处理完玩家选择
                        )
                        action_type = director_decision.get("action_type", "继续发言")
                        
                        if action_type == "切换节点":
                            next_node = node_config.get("下一节点")
                            if next_node:
                                nodes = scene_config.get("节点列表", [])
                                next_idx = next(
                                    (i for i, n in enumerate(nodes) if n.get("节点ID") == next_node),
                                    self.current_node_idx + 1
                                )
                                await self._enter_node(scene_config, next_idx)
                                return
                            else:
                                await self._check_scene_end(scene_config)
                                return
                        elif action_type == "结束场景":
                            await self._check_scene_end(scene_config)
                            return
                        elif action_type == "玩家选项":
                            # 如果刚刚处理完玩家选择，不应该立即再次显示
                            # 改为继续发言，让其他角色也参与
                            print(f"[导演] 刚刚处理完玩家选择，改为继续角色对话")
                            action_type = "继续发言"
                            next_speaker = director_decision.get("next_speaker")
                            if not next_speaker or next_speaker == "玩家选项" or next_speaker not in roles:
                                # 选择其他角色继续对话
                                remaining_roles = [r for r in roles if r != trigger_role]
                                next_speaker = remaining_roles[0] if remaining_roles else roles[0]
                        elif action_type == "继续发言":
                            next_speaker = director_decision.get("next_speaker")
                            if not next_speaker or (next_speaker != "玩家选项" and next_speaker not in roles):
                                break
                    else:
                        print(f"[警告] 角色 '{trigger_role}' 不存在，跳过回应")
                        # 继续由导演决定下一步
                        director_decision = self.director_agent.decide_next_action(
                            current_state=current_state,
                            last_speaker=None,
                            last_speech=None
                        )
                        action_type = director_decision.get("action_type", "继续发言")
                        if action_type == "切换节点" or action_type == "结束场景":
                            if action_type == "切换节点":
                                next_node = node_config.get("下一节点")
                                if next_node:
                                    nodes = scene_config.get("节点列表", [])
                                    next_idx = next(
                                        (i for i, n in enumerate(nodes) if n.get("节点ID") == next_node),
                                        self.current_node_idx + 1
                                    )
                                    await self._enter_node(scene_config, next_idx)
                                    return
                            await self._check_scene_end(scene_config)
                            return
                        next_speaker = director_decision.get("next_speaker")
                else:
                    print(f"[导演] 玩家选择'{self.player_choice}'，无需角色回应")
                    # 继续由导演决定下一步
                    director_decision = self.director_agent.decide_next_action(
                        current_state=current_state,
                        last_speaker=None,
                        last_speech=None
                    )
                    action_type = director_decision.get("action_type", "继续发言")
                    if action_type == "切换节点" or action_type == "结束场景":
                        if action_type == "切换节点":
                            next_node = node_config.get("下一节点")
                            if next_node:
                                nodes = scene_config.get("节点列表", [])
                                next_idx = next(
                                    (i for i, n in enumerate(nodes) if n.get("节点ID") == next_node),
                                    self.current_node_idx + 1
                                )
                                await self._enter_node(scene_config, next_idx)
                                return
                        await self._check_scene_end(scene_config)
                        return
                    next_speaker = director_decision.get("next_speaker")
            
            # 如果当前发言者是角色，生成发言
            elif next_speaker in self.role_agents:
                speech_result = await self._generate_role_speech(next_speaker, node_config)
                
                # 更新状态
                current_state = {
                    "scene": self.system_agent.current_scene,
                    "node": node_config.get("节点ID"),
                    "global_state": self.system_agent.global_state
                }
                
                # 记录发言
                speech_text = ""
                if speech_result:
                    speech_text = speech_result.get("dialogue", "") + " " + speech_result.get("description", "")
                
                self.director_agent.add_speech_record(next_speaker, speech_text)
                
                # 发言后，由导演决定下一步
                director_decision = self.director_agent.decide_next_action(
                    current_state=current_state,
                    last_speaker=next_speaker,
                    last_speech=speech_text
                )
                
                # 根据导演决策执行下一步
                action_type = director_decision.get("action_type", "继续发言")
                
                if action_type == "切换节点":
                    # 切换到下一节点
                    next_node = node_config.get("下一节点")
                    if next_node:
                        nodes = scene_config.get("节点列表", [])
                        next_idx = next(
                            (i for i, n in enumerate(nodes) if n.get("节点ID") == next_node),
                            self.current_node_idx + 1
                        )
                        await self._enter_node(scene_config, next_idx)
                        return  # 已切换节点，退出当前执行
                    else:
                        # 节点结束，检查场景
                        await self._check_scene_end(scene_config)
                        return
                
                elif action_type == "结束场景":
                    # 结束当前场景
                    await self._check_scene_end(scene_config)
                    return
                
                elif action_type == "玩家选项":
                    # 导演决定需要玩家选项
                    next_speaker = "玩家选项"
                    # 继续循环处理玩家选项
                
                elif action_type == "继续发言":
                    # 导演指定下一个发言者
                    next_speaker = director_decision.get("next_speaker")
                    if not next_speaker or (next_speaker != "玩家选项" and next_speaker not in roles):
                        # 如果没有指定或无效，结束节点
                        break
                    
                    # 如果有表演指导，传递给角色Agent（在下次生成时使用）
                    acting_guidance = director_decision.get("acting_guidance")
                    if acting_guidance and acting_guidance.get("role") in self.role_agents:
                        # 保存指导，供下次生成使用
                        role_agent = self.role_agents[acting_guidance["role"]]
                        role_agent.set_acting_guidance(acting_guidance)
            else:
                # 无效的发言者，结束
                print(f"[警告] 无效的发言者 '{next_speaker}'，结束节点")
                break
        
        # 所有发言完成后，检查推进条件
        advance_condition = node_config.get("推进条件", "")
        if "所有角色发言完成" in advance_condition:
            # 进入下一节点
            next_node = node_config.get("下一节点")
            if next_node:
                # 找到下一节点索引
                nodes = scene_config.get("节点列表", [])
                next_idx = next(
                    (i for i, n in enumerate(nodes) if n.get("节点ID") == next_node),
                    self.current_node_idx + 1
                )
                await self._enter_node(scene_config, next_idx)
            else:
                # 节点结束
                await self._check_scene_end(scene_config)
    
    async def _generate_role_speech(
        self,
        role_id: str,
        node_config: Dict,
        response_strategy: Optional[str] = None
    ) -> Optional[Dict[str, str]]:
        """生成角色发言"""
        if role_id not in self.role_agents:
            print(f"[警告] 角色 {role_id} 不存在")
            return None
        
        # 添加分隔线，使输出更清晰
        print(f"\n{'='*60}")
        
        role_agent = self.role_agents[role_id]
        
        # 获取上下文
        context = f"当前剧情节点：{node_config.get('节点ID', '')}"
        memory_context = self.system_agent.memory_system.get_context_text(
            role=role_id,
            scene=self.system_agent.current_scene
        )
        
        # 如果有响应策略，添加到上下文
        if response_strategy:
            context += f"\n响应策略：{response_strategy}"
        
        # 生成发言
        result = await role_agent.generate_speech(context, node_config, memory_context)
        
        # 显示发言（改进格式）
        description = result.get("description", "").strip()
        dialogue = result.get("dialogue", "").strip()
        
        # 清理格式：移除多余的括号和标记
        if description.startswith("（") and description.endswith("）"):
            description = description[1:-1]
        if description.startswith("(") and description.endswith(")"):
            description = description[1:-1]
        
        # 输出格式：先描述，后对话
        # 输出角色发言（改进格式，确保清晰分离）
        if description:
            # 清理描述中的格式标记
            description = description.replace("描述：", "").replace("描述", "").strip()
            # 移除多余的括号
            if description.startswith("（") and description.endswith("）"):
                description = description[1:-1].strip()
            if description.startswith("(") and description.endswith(")"):
                description = description[1:-1].strip()
            # 如果描述只有角色名，使用默认描述
            if description.strip() in [role_agent.role_name, role_agent.role_id]:
                description = "看向众人"
            print(f"[{role_agent.role_name}] {description}")
        
        if dialogue:
            # 清理对话中的格式标记
            dialogue = dialogue.replace("对话：", "").replace("对话", "").strip()
            # 移除引号标记（如果有）
            dialogue = dialogue.strip('"').strip('"').strip("'").strip("'")
            # 移除描述标记（如果混在对话中）
            import re
            dialogue = re.sub(r'描述[：:]?\s*[（(].+?[）)]', '', dialogue).strip()
            dialogue = re.sub(r'\s*描述[：:]?\s*', '', dialogue).strip()
            print(f"[{role_agent.role_name}] \"{dialogue}\"")
        
        # 如果两者都为空，输出警告
        if not description and not dialogue:
            print(f"[警告] 角色 {role_agent.role_name} 的发言为空，使用默认输出")
            print(f"[{role_agent.role_name}] 看向众人")
            print(f"[{role_agent.role_name}] \"...\"")
        
        print()  # 空行分隔
        
        # 保存到记忆
        self.system_agent.memory_system.add_memory(
            content=f"{dialogue}",
            memory_type="角色发言",
            role=role_id,
            scene=self.system_agent.current_scene,
            importance=0.5
        )
        
        # 返回结果供导演Agent使用
        return {
            "dialogue": dialogue,
            "description": description
        }
    
    async def _check_scene_end(self, scene_config: Dict):
        """检查场景结束"""
        if self.director_agent.check_scene_end_condition():
            # 场景结束，压缩记忆
            self.system_agent.memory_system.compress_memories(
                current_scene=self.system_agent.current_scene
            )
            
            # 卸载当前场景角色LoRA
            role_list = scene_config.get("角色列表", [])
            for role_id in role_list:
                self.system_agent.unload_lora(role_id)
            
            # 切换到下一场景
            next_scene = self.director_agent.get_next_scene()
            if next_scene:
                # 找到下一场景索引
                scenes = self.scenario_config["剧本"]["场景列表"]
                next_idx = next(
                    (i for i, s in enumerate(scenes) if s.get("场景ID") == next_scene),
                    self.current_scene_idx + 1
                )
                self.current_scene_idx = next_idx
                await self._enter_scene(next_idx)
            else:
                print("\n剧情结束")
        else:
            # 继续下一节点
            await self._enter_node(scene_config, self.current_node_idx + 1)


async def main():
    """主函数"""
    # 使用相对于脚本文件的路径
    script_dir = Path(__file__).parent
    scenario_path = script_dir / "config" / "example_scenario.yaml"
    characters_path = script_dir / "config" / "example_characters.yaml"
    
    engine = DramaEngine(
        scenario_path=str(scenario_path),
        characters_path=str(characters_path)
    )
    
    await engine.start()


if __name__ == "__main__":
    asyncio.run(main())
