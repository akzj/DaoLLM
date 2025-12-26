# 训练显存问题修复指南

## 问题分析

训练开始时遇到了显存不足错误：
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.32 GiB.
```

**原因**：
- Loss计算时需要将logits转换为float32，需要额外显存
- Batch size=2 + max_seq_length=2048 对16GB显存来说太大
- 需要启用gradient checkpointing等优化

## 已应用的修复

### 1. 启用Gradient Checkpointing
- 在模型加载后自动启用
- 可以节省约30-50%的显存

### 2. 优化训练参数
- `gradient_checkpointing=True`
- `dataloader_pin_memory=False`
- `max_grad_norm=1.0`

### 3. 降低默认参数
- `max_seq_length`: 2048 → 1024
- 减少显存占用

## 推荐的训练参数

### 方案1：保守设置（推荐，确保能运行）

```bash
cd fine-tune
source venv/bin/activate

python train_lora.py \
    --model_name_or_path Qwen/Qwen1.5-7B-Chat \
    --data_path ./data/dataset \
    --output_dir ./output \
    --use_4bit \
    --bf16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-4 \
    --lora_rank 32 \
    --lora_alpha 64 \
    --save_steps 500 \
    --max_seq_length 1024 \
    --logging_steps 10 \
    --dataloader_num_workers 2
```

**显存占用预估**：~10-12GB

### 方案2：使用修复脚本

```bash
cd fine-tune
./train_lora_fixed.sh
```

### 方案3：如果仍然OOM，进一步降低

```bash
python train_lora.py \
    --model_name_or_path Qwen/Qwen1.5-7B-Chat \
    --data_path ./data/dataset \
    --output_dir ./output \
    --use_4bit \
    --bf16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --learning_rate 2e-4 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --save_steps 500 \
    --max_seq_length 512 \
    --logging_steps 10 \
    --dataloader_num_workers 1
```

**显存占用预估**：~8-10GB

## 参数说明

| 参数 | 原值 | 推荐值 | 说明 |
|------|------|--------|------|
| `per_device_train_batch_size` | 2 | 1 | 减少单次batch显存占用 |
| `gradient_accumulation_steps` | 8 | 16-32 | 保持有效batch size |
| `max_seq_length` | 2048 | 1024-512 | 减少序列长度显存占用 |
| `lora_rank` | 64 | 32-16 | 减少LoRA参数量 |
| `dataloader_num_workers` | 4 | 2-1 | 减少CPU内存占用 |

## 显存优化技巧

1. **Gradient Checkpointing**：已自动启用
2. **4bit量化**：已启用
3. **BF16混合精度**：已启用
4. **减少序列长度**：降低max_seq_length
5. **减少batch size**：使用gradient accumulation补偿

## 监控训练

训练开始后，使用以下命令监控：

```bash
# 实时监控GPU
watch -n 1 nvidia-smi

# 检查训练进度
python check_training_progress.py
```

## 预期显存使用

- **模型加载**：~6-7GB（4bit量化）
- **训练时**：~10-12GB（batch_size=1, seq_len=1024）
- **峰值**：~12-14GB（梯度计算时）

## 如果仍然OOM

1. 检查是否有其他进程占用显存：`nvidia-smi`
2. 进一步降低max_seq_length到512
3. 降低lora_rank到16
4. 增加gradient_accumulation_steps到32

