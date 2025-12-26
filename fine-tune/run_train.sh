#!/bin/bash
# LoRA微调快速启动脚本

# 设置默认参数
MODEL_NAME="Qwen/Qwen1.5-7B-Chat"
DATA_PATH="./data/dataset"
OUTPUT_DIR="./output"
NUM_EPOCHS=3
BATCH_SIZE=2
GRAD_ACCUM=8
LEARNING_RATE=2e-4
LORA_RANK=64
LORA_ALPHA=128

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --grad_accum)
            GRAD_ACCUM="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --lora_rank)
            LORA_RANK="$2"
            shift 2
            ;;
        --lora_alpha)
            LORA_ALPHA="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行训练
python train_lora.py \
    --model_name_or_path "$MODEL_NAME" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs "$NUM_EPOCHS" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --learning_rate "$LEARNING_RATE" \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --use_4bit \
    --bf16

