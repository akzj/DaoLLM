#!/bin/bash
# 低显存训练脚本 - 适用于显存受限的情况

# 设置环境变量避免显存碎片化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 检查是否有其他进程占用显存
echo "检查GPU显存使用情况..."
nvidia-smi

echo ""
echo "如果看到有其他Python进程占用显存，请先关闭它们"
echo "按Enter继续，或Ctrl+C取消..."
read

# 清理显存
python -c "import torch; torch.cuda.empty_cache(); print('显存缓存已清理')"

# 使用更小的batch size和gradient accumulation
cd /home/ubuntu/workspace/DaoLLM/fine-tune
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
    --dataloader_num_workers 2

