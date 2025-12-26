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
        self.director_agent = DirectorAgent(self.system_agent)
        
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
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    async def start(self):
        """开始剧情演绎"""
        print("=" * 60)
        print("修仙世界剧情演绎开始")
        print("=" * 60)
        
        # 进入第一个场景
        await self._enter_scene(0)
    
    async def _enter_scene(self, scene_idx: int):
        """进入场景"""
        scenes = self.scenario_config["剧本"]["场景列表"]
        if scene_idx >= len(scenes):
            print("剧情结束")
            return
        
        scene_config = scenes[scene_idx]
        scene_id = scene_config["场景ID"]
        
        print(f"\n[场景切换] 进入场景: {scene_id}")
        
        # 生成场景描述
        scene_description = self.system_agent.generate_scene_description(scene_config)
        print(f"\n{scene_description}\n")
        
        # 保存场景描述到记忆
        self.system_agent.memory_system.add_memory(
            content=f"进入场景：{scene_id}。{scene_description}",
            memory_type="剧情事件",
            scene=scene_id,
            importance=1.0
        )
        
        # 设置当前场景
        self.system_agent.current_scene = scene_id
        self.director_agent.set_scene(scene_config)
        
        # 加载场景角色LoRA
        role_list = scene_config.get("角色列表", [])
        lora_loaded_count = 0
        lora_missing_count = 0
        
        for role_id in role_list:
            if role_id in self.role_agents:
                role_config = next(
                    (r for r in self.characters_config["角色"] if r["角色ID"] == role_id),
                    None
                )
                if role_config:
                    lora_path = role_config.get("LoRA路径", "")
                    lora_model = self.system_agent.load_lora(role_id, lora_path, verbose=True)
                    if lora_model:
                        lora_loaded_count += 1
                    elif lora_path and lora_path.strip():
                        lora_missing_count += 1
        
        # 输出LoRA加载总结
        if lora_loaded_count > 0:
            print(f"[LoRA] 成功加载 {lora_loaded_count} 个角色LoRA")
        if lora_missing_count > 0:
            print(f"[提示] {lora_missing_count} 个角色LoRA未找到，将使用基础模型")
        if lora_loaded_count == 0 and lora_missing_count == 0:
            print(f"[提示] 当前场景角色均未配置LoRA，使用基础模型")
        
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
        """执行节点"""
        # 决策发言顺序
        current_state = {
            "scene": self.system_agent.current_scene,
            "node": node_config.get("节点ID"),
            "global_state": self.system_agent.global_state
        }
        
        decision = self.director_agent.decide_speech_order(current_state, self.player_choice)
        speech_order = decision["speech_order"]
        player_response = decision.get("player_response")
        
        # 执行发言顺序
        for speaker in speech_order:
            if speaker == "玩家选项":
                # 生成玩家选项
                context = self.system_agent.memory_system.get_context_text(
                    scene=self.system_agent.current_scene
                )
                options = self.system_agent.generate_player_options(node_config, context)
                
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
                
                # 如果有玩家响应角色，生成响应
                if player_response:
                    trigger_role = player_response["trigger_role"]
                    if trigger_role in self.role_agents:
                        await self._generate_role_speech(
                            trigger_role,
                            node_config,
                            player_response["response_strategy"]
                        )
            
            elif speaker == "玩家选择":
                # 跳过，已经在"玩家选项"中处理
                continue
            
            else:
                # 角色发言
                await self._generate_role_speech(speaker, node_config)
        
        # 检查推进条件
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
    ):
        """生成角色发言"""
        if role_id not in self.role_agents:
            print(f"[警告] 角色 {role_id} 不存在")
            return
        
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
        
        # 显示发言
        description = result.get("description", "")
        dialogue = result.get("dialogue", "")
        
        if description:
            print(f"[{role_agent.role_name}] {description}")
        print(f"[{role_agent.role_name}] {dialogue}\n")
        
        # 保存到记忆
        self.system_agent.memory_system.add_memory(
            content=f"{dialogue}",
            memory_type="角色发言",
            role=role_id,
            scene=self.system_agent.current_scene,
            importance=0.5
        )
    
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

