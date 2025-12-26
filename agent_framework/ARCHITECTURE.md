# 架构设计文档

## 项目结构

```
agent_framework/
├── __init__.py              # 包初始化
├── main.py                  # 主流程控制（DramaEngine）
├── run.py                   # 运行脚本
├── requirements.txt         # 依赖列表
├── README.md               # 使用说明
├── ARCHITECTURE.md         # 架构文档（本文件）
├── agents/                  # Agent模块
│   ├── __init__.py
│   ├── system_agent.py     # 系统Agent（资源管理）
│   ├── director_agent.py   # 导演Agent（剧情协调）
│   └── role_agent.py       # 角色Agent（发言生成）
├── memory/                  # 记忆系统
│   ├── __init__.py
│   └── memory_system.py    # 记忆存储和压缩
├── config/                  # 配置文件
│   ├── example_scenario.yaml    # 剧本配置示例
│   └── example_characters.yaml  # 角色配置示例
└── utils/                   # 工具函数
    ├── __init__.py
    └── config_loader.py     # 配置加载工具
```

## 核心组件

### 1. SystemAgent（系统Agent）

**职责**：
- 基础模型和LoRA权重管理
- 场景描述生成
- 玩家选项生成
- 全局状态管理
- 资源锁定（LoRA锁）

**关键方法**：
- `load_lora()`: 动态加载LoRA权重
- `unload_lora()`: 卸载LoRA权重（释放显存）
- `generate_scene_description()`: 生成场景描述
- `generate_player_options()`: 生成玩家选项

### 2. DirectorAgent（导演Agent）

**职责**：
- 决策角色发言顺序（LLM辅助）
- 决策玩家选择的响应角色和策略
- 判断场景切换条件
- 协调剧情推进

**关键方法**：
- `decide_speech_order()`: 决策发言顺序
- `check_scene_end_condition()`: 检查场景结束条件
- `get_next_scene()`: 获取下一场景

### 3. RoleAgent（角色Agent）

**职责**：
- 生成角色发言（对话+行为描述）
- 管理角色LoRA权重
- 异步生成执行

**关键方法**：
- `generate_speech()`: 异步生成角色发言
- `_generate_with_lock()`: 使用LoRA锁的同步生成方法

### 4. MemorySystem（记忆系统）

**职责**：
- 记忆存储和检索
- 混合压缩策略执行
- 上下文管理

**压缩策略**：
1. **重要性排序**：基于规则计算重要性分数
2. **语义相似度去重**：使用sentence-transformers计算相似度
3. **时间衰减**：根据时间差和重要性应用衰减
4. **Token截断**：控制总token数≤1024

**关键方法**：
- `add_memory()`: 添加记忆
- `compress_memories()`: 压缩记忆
- `get_context_text()`: 获取上下文文本

### 5. DramaEngine（剧情引擎）

**职责**：
- 主流程控制
- 场景和节点管理
- 玩家交互处理
- 协调各Agent协作

**关键流程**：
1. 加载配置（剧本+角色）
2. 初始化Agent
3. 进入场景 → 生成场景描述
4. 进入节点 → 决策发言顺序
5. 执行发言 → 角色异步生成
6. 检查推进条件 → 切换节点/场景
7. 记忆压缩和资源清理

## 数据流

```
玩家输入
  ↓
SystemAgent解析意图
  ↓
DirectorAgent决策发言顺序
  ↓
RoleAgent异步生成回复（并行）
  ↓
MemorySystem保存记忆
  ↓
DirectorAgent纠偏（可选）
  ↓
SystemAgent更新全局状态
  ↓
检查剧情结束条件
```

## 资源管理

### 显存优化

1. **4bit量化**：基础模型使用4bit量化
2. **动态LoRA加载**：按场景加载/卸载LoRA
3. **记忆压缩**：控制上下文token数≤1024
4. **异步生成**：使用线程池避免阻塞

### 并发控制

- **LoRA锁**：确保同一时间只有一个Agent使用LoRA模型
- **异步生成**：角色发言使用`asyncio.run_in_executor`在线程池执行

## 配置格式

### 剧本配置（YAML）

```yaml
剧本:
  元信息:
    标题: "..."
    描述: "..."
  
  场景列表:
    - 场景ID: "..."
      场景描述: "..."
      角色列表: ["角色1", "角色2"]
      节点列表:
        - 节点ID: "..."
          发言顺序: ["角色1", "玩家选项", "角色2"]
          玩家选项: ["选项1", "选项2"]
          推进条件: "..."
          下一节点: "..."
```

### 角色配置（YAML）

```yaml
角色:
  - 角色ID: "..."
    名称: "..."
    人设: |
      ...
    LoRA路径: "./lora/角色名"
    初始目标: "..."
```

## 扩展点

1. **玩家输入处理**：当前简化处理，可扩展为异步输入
2. **LLM决策优化**：导演Agent的决策逻辑可进一步优化
3. **记忆重要性计算**：可引入更复杂的规则或模型
4. **多场景并行**：当前单场景，可扩展为多场景切换
5. **可视化界面**：可添加Web界面或GUI

## 性能考虑

- **显存占用**：峰值≤9.5GB（4070 Ti 12GB）
- **生成速度**：异步生成，避免阻塞
- **记忆压缩**：定期压缩，控制token数
- **LoRA切换**：按需加载，及时卸载

