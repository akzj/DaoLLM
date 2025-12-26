#!/bin/bash
# 修复显存问题的训练脚本

# 设置环境变量避免显存碎片化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/ubuntu/workspace/DaoLLM/fine-tune
source venv/bin/activate

# 清理显存
python -c "import torch; torch.cuda.empty_cache(); print('显存缓存已清理')"

echo "开始训练（使用优化的显存设置）..."
echo "参数："
echo "  - Batch size: 1"
echo "  - Gradient accumulation: 16"
echo "  - Max seq length: 1024"
echo "  - LoRA rank: 32"
echo ""

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

