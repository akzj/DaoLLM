# 修仙世界AI Agent框架

基于多Agent协同的修仙世界剧情演绎系统，实现沉浸式文字舞台剧体验。

## 核心特性

- **多Agent协同**：系统Agent、导演Agent、角色Agent分工明确
- **LoRA微调**：每个角色使用独立的LoRA权重，贴合人设
- **记忆系统**：混合压缩策略（重要性排序+语义相似度去重+时间衰减）
- **异步生成**：角色发言异步生成，避免阻塞
- **资源管理**：4bit量化，动态LoRA加载，控制显存占用

## 架构设计

### 核心组件

1. **SystemAgent（系统Agent）**
   - 基础模型和LoRA管理
   - 场景描述生成
   - 玩家选项生成
   - 全局状态管理

2. **DirectorAgent（导演Agent）**
   - 发言顺序决策（LLM辅助）
   - 玩家选择响应决策
   - 场景切换判断

3. **RoleAgent（角色Agent）**
   - 角色发言生成（对话+描述）
   - LoRA权重加载
   - 异步生成执行

4. **MemorySystem（记忆系统）**
   - 记忆存储和检索
   - 混合压缩策略
   - 上下文管理

## 配置格式

### 剧本配置（YAML）

```yaml
剧本:
  元信息:
    标题: "秘境试炼"
    描述: "..."
  
  场景列表:
    - 场景ID: "客栈休整"
      场景描述: "..."
      角色列表: ["角色A", "角色B"]
      节点列表: [...]
```

### 角色配置（YAML）

```yaml
角色:
  - 角色ID: "角色A"
    名称: "李师兄"
    人设: "..."
    LoRA路径: "./lora/角色A"
```

## 使用方法

### 1. 安装依赖

```bash
cd agent_framework
pip install -r requirements.txt
```

### 2. 准备配置文件

- 准备剧本配置文件（YAML格式），参考 `config/example_scenario.yaml`
- 准备角色配置文件（YAML格式），参考 `config/example_characters.yaml`
- 准备LoRA权重（可选，如无则使用基础模型）

### 3. 运行主程序

使用运行脚本（推荐）：
```bash
cd agent_framework
python run.py
```

或使用自定义配置：
```bash
python run.py --scenario ./config/your_scenario.yaml --characters ./config/your_characters.yaml
```

**注意**：玩家输入时，请输入选项编号（如 `1`、`2`）或选项内容，不要输入shell命令。详见 `QUICKSTART.md`。

### 4. 运行测试

```bash
python test_simple.py
```

## 命令行参数

`run.py` 支持以下参数：

- `--scenario`: 剧本配置文件路径（默认：`./config/example_scenario.yaml`）
- `--characters`: 角色配置文件路径（默认：`./config/example_characters.yaml`）
- `--model`: 基础模型名称或路径（默认：`Qwen/Qwen1.5-7B-Chat`）

## 项目结构

```
agent_framework/
├── agents/              # Agent模块
│   ├── system_agent.py  # 系统Agent
│   ├── director_agent.py # 导演Agent
│   └── role_agent.py    # 角色Agent
├── memory/              # 记忆系统
│   └── memory_system.py
├── config/             # 配置文件
│   ├── example_scenario.yaml
│   └── example_characters.yaml
├── utils/              # 工具函数
│   ├── player_input.py  # 玩家输入处理
│   ├── logger.py        # 日志工具
│   ├── error_handler.py  # 错误处理
│   ├── config_validator.py # 配置验证
│   └── performance_monitor.py # 性能监控
├── main.py             # 主程序
├── run.py              # 运行脚本
└── test_simple.py      # 测试脚本
```

## 注意事项

- **硬件要求**：需要12GB+显存（推荐4070 Ti或更高）
- **模型下载**：首次运行需要下载模型和embedding模型（sentence-transformers）
- **LoRA权重**：LoRA路径如果不存在，会使用基础模型
- **配置验证**：程序启动时会自动验证配置文件格式
- **异步输入**：玩家输入使用异步处理，支持超时和取消

## 开发说明

### 扩展点

1. **玩家输入**：可在 `utils/player_input.py` 中扩展输入方式（如GUI、Web接口）
2. **记忆压缩**：可在 `memory/memory_system.py` 中调整压缩策略
3. **生成Prompt**：可在各Agent中优化Prompt模板
4. **性能监控**：使用 `utils/performance_monitor.py` 监控性能指标

### 调试技巧

- 使用 `test_simple.py` 测试基本功能
- 检查配置文件格式是否正确
- 查看控制台输出的错误信息
- 使用性能监控工具查看资源使用情况

