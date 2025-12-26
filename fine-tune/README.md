# Qwen1.5-7B-Chat LoRA微调

本项目用于对Qwen1.5-7B-Chat模型进行LoRA微调，专门针对修仙背景世界的AI游戏场景。

## 环境要求

- Python 3.8+
- CUDA 11.8+ (推荐)
- 至少16GB显存 (使用4bit量化时)

## 安装依赖

```bash
pip install -r requirements.txt
```

## 数据准备

### 从原始文本生成数据集

如果 `data/dataset/` 目录不存在或需要重新生成，可以使用 `prepare_data.py` 脚本从原始小说文本生成训练数据集：

```bash
python prepare_data.py \
    --novels_dir ./data/novels_raw \
    --output_dir ./data/dataset \
    --model_name Qwen/Qwen1.5-7B-Chat \
    --max_length 2048 \
    --stride 512
```

参数说明：
- `--novels_dir`: 原始小说文本目录，默认为 `./data/novels_raw`
- `--output_dir`: 输出数据集目录，默认为 `./data/dataset`
- `--model_name`: 模型名称（用于tokenizer），默认为 `Qwen/Qwen1.5-7B-Chat`
- `--max_length`: 最大序列长度，默认为 2048
- `--stride`: 滑动窗口步长，默认为 512

脚本会：
1. 读取 `novels_raw/` 目录下的所有 `.txt` 文件
2. 使用 Qwen tokenizer 进行 tokenization
3. 使用滑动窗口切分成固定长度的序列
4. 生成 Arrow 格式数据集并保存到 `data/dataset/`

### 数据集格式

生成的数据集包含两个字段：
- `input_ids`: tokenized 的输入序列（int32）
- `labels`: 标签序列（int64），与 input_ids 相同（用于 causal LM）

## 使用方法

### 基本使用

```bash
python train_lora.py \
    --model_name_or_path Qwen/Qwen1.5-7B-Chat \
    --data_path ./data/dataset \
    --output_dir ./output \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --use_4bit
```

### 参数说明

#### 模型参数
- `--model_name_or_path`: 预训练模型路径，默认为 `Qwen/Qwen1.5-7B-Chat`
- `--trust_remote_code`: 信任远程代码（默认开启）

#### 数据参数
- `--data_path`: 数据集路径，默认为 `./data/dataset`
- `--max_seq_length`: 最大序列长度，默认为 2048

#### LoRA参数
- `--lora_rank`: LoRA rank，默认为 64
- `--lora_alpha`: LoRA alpha，默认为 128
- `--lora_dropout`: LoRA dropout，默认为 0.1
- `--use_4bit`: 使用4bit量化（默认开启）

#### 训练参数
- `--output_dir`: 输出目录，默认为 `./output`
- `--num_train_epochs`: 训练轮数，默认为 3
- `--per_device_train_batch_size`: 每设备训练批次大小，默认为 2
- `--gradient_accumulation_steps`: 梯度累积步数，默认为 8
- `--learning_rate`: 学习率，默认为 2e-4
- `--warmup_steps`: 预热步数，默认为 100
- `--logging_steps`: 日志记录步数，默认为 10
- `--save_steps`: 保存步数，默认为 500
- `--bf16`: 使用bf16精度（默认开启）

### 使用快速启动脚本

也可以使用提供的shell脚本快速启动训练：

```bash
./run_train.sh
```

或者自定义参数：

```bash
./run_train.sh \
    --model_name Qwen/Qwen1.5-7B-Chat \
    --data_path ./data/dataset \
    --output_dir ./output \
    --num_epochs 3 \
    --batch_size 2 \
    --grad_accum 8 \
    --learning_rate 2e-4 \
    --lora_rank 64 \
    --lora_alpha 128
```

### 使用配置文件

可以使用 `config.yaml` 配置文件，但当前版本需要通过命令行参数传递。

## 输出

训练完成后，模型将保存在 `output_dir` 指定的目录下，包括：
- LoRA适配器权重
- Tokenizer文件
- 训练日志（TensorBoard格式）

## 推理测试

### 使用推理脚本

训练完成后，可以使用 `inference.py` 脚本测试模型：

#### 单次推理

```bash
python inference.py \
    --base_model Qwen/Qwen1.5-7B-Chat \
    --lora_model ./output \
    --prompt "你好，请介绍一下修仙世界" \
    --use_4bit
```

#### 交互模式

```bash
python inference.py \
    --base_model Qwen/Qwen1.5-7B-Chat \
    --lora_model ./output \
    --interactive \
    --use_4bit
```

### 在代码中使用

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-7B-Chat",
    trust_remote_code=True,
    device_map="auto"
)

# 加载LoRA权重
model = PeftModel.from_pretrained(base_model, "./output")

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained("./output")

# 使用模型（Qwen1.5-Chat格式）
messages = [{"role": "user", "content": "你好，请介绍一下修仙世界"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)
response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(response)
```

## 注意事项

1. **显存要求**: 使用4bit量化时，7B模型大约需要12-16GB显存
2. **数据格式**: 确保数据集已经正确预处理，包含 `input_ids` 和 `labels` 字段
3. **训练时间**: 根据数据量大小，训练可能需要数小时到数天
4. **检查点**: 训练过程中会定期保存检查点，可以从中断处继续训练

## 故障排除

### 显存不足
- 减小 `per_device_train_batch_size`
- 增大 `gradient_accumulation_steps`
- 确保使用 `--use_4bit` 参数

### 训练速度慢
- 检查CUDA是否正确安装
- 使用 `bf16` 或 `fp16` 加速训练
- 调整 `dataloader_num_workers`

## 许可证

MIT License

