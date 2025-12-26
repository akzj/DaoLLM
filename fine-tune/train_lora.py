#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA微调脚本 for Qwen1.5-7B-Chat
用于修仙背景世界的AI游戏
"""

import os
import torch
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """模型相关参数"""
    model_name_or_path: str = field(
        default="Qwen/Qwen1.5-7B-Chat",
        metadata={"help": "预训练模型路径"}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "是否信任远程代码"}
    )


@dataclass
class DataArguments:
    """数据相关参数"""
    data_path: str = field(
        default="./data/dataset",
        metadata={"help": "数据集路径"}
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "最大序列长度"}
    )


@dataclass
class LoraArguments:
    """LoRA相关参数"""
    lora_rank: int = field(
        default=64,
        metadata={"help": "LoRA rank"}
    )
    lora_alpha: int = field(
        default=128,
        metadata={"help": "LoRA alpha"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA dropout"}
    )
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        metadata={"help": "LoRA目标模块"}
    )
    use_4bit: bool = field(
        default=True,
        metadata={"help": "是否使用4bit量化"}
    )


def load_model_and_tokenizer(
    model_args: ModelArguments,
    lora_args: LoraArguments
):
    """加载模型和tokenizer"""
    logger.info(f"加载模型: {model_args.model_name_or_path}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="right",
    )
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 配置量化（如果使用）
    if lora_args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None
    
    # 加载模型
    # 使用low_cpu_mem_usage减少CPU内存占用
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if not lora_args.use_4bit else None,
        low_cpu_mem_usage=True,
    )
    
    # 清理显存缓存
    torch.cuda.empty_cache()
    
    # 清理显存缓存
    torch.cuda.empty_cache()
    
    # 准备模型用于k-bit训练
    if lora_args.use_4bit:
        # 设置环境变量避免显存碎片化
        import os
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        model = prepare_model_for_kbit_training(model)
        
        # 再次清理显存
        torch.cuda.empty_cache()
    
    # 配置LoRA
    lora_config = LoraConfig(
        r=lora_args.lora_rank,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # 应用LoRA
    model = get_peft_model(model, lora_config)
    
    # 启用gradient checkpointing以节省显存
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("已启用gradient checkpointing以节省显存")
    
    # 打印可训练参数
    model.print_trainable_parameters()
    
    return model, tokenizer


def load_dataset(data_args: DataArguments):
    """加载数据集"""
    logger.info(f"加载数据集: {data_args.data_path}")
    logger.info("正在从磁盘加载数据集，这可能需要一些时间...")
    dataset = load_from_disk(data_args.data_path)
    logger.info(f"数据集加载完成，样本数: {len(dataset) if hasattr(dataset, '__len__') else '未知'}")
    return dataset


def preprocess_function(examples, tokenizer, max_length):
    """预处理函数"""
    # 如果数据已经是tokenized的，直接返回
    if "input_ids" in examples:
        return examples
    
    # 否则需要tokenize
    # 这里假设数据是文本格式
    texts = examples.get("text", examples.get("content", []))
    if not texts:
        return examples
    
    # Tokenize
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None,
    )
    
    # 设置labels
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LoRA微调Qwen1.5-7B-Chat")
    
    # 模型参数
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen1.5-7B-Chat",
                       help="预训练模型路径")
    parser.add_argument("--trust_remote_code", action="store_true", default=True,
                       help="信任远程代码")
    
    # 数据参数
    parser.add_argument("--data_path", type=str, default="./data/dataset",
                       help="数据集路径")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                       help="最大序列长度")
    
    # LoRA参数
    parser.add_argument("--lora_rank", type=int, default=64,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=128,
                       help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                       help="LoRA dropout")
    parser.add_argument("--use_4bit", action="store_true", default=True,
                       help="使用4bit量化")
    
    # 训练参数
    parser.add_argument("--output_dir", type=str, default="./output",
                       help="输出目录")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                       help="训练轮数")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2,
                       help="每设备训练批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                       help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                       help="学习率")
    parser.add_argument("--warmup_steps", type=int, default=100,
                       help="预热步数")
    parser.add_argument("--logging_steps", type=int, default=10,
                       help="日志记录步数")
    parser.add_argument("--save_steps", type=int, default=500,
                       help="保存步数")
    parser.add_argument("--save_total_limit", type=int, default=3,
                       help="保存检查点数量限制")
    parser.add_argument("--fp16", action="store_true", default=False,
                       help="使用fp16")
    parser.add_argument("--bf16", action="store_true", default=True,
                       help="使用bf16")
    parser.add_argument("--dataloader_num_workers", type=int, default=4,
                       help="数据加载器工作进程数")
    parser.add_argument("--remove_unused_columns", action="store_true", default=False,
                       help="移除未使用的列")
    
    args = parser.parse_args()
    
    # 创建参数对象
    model_args = ModelArguments(
        model_name_or_path=args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )
    
    data_args = DataArguments(
        data_path=args.data_path,
        max_seq_length=args.max_seq_length,
    )
    
    lora_args = LoraArguments(
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_4bit=args.use_4bit,
    )
    
    # 加载模型和tokenizer
    model, tokenizer = load_model_and_tokenizer(model_args, lora_args)
    
    # 加载数据集
    logger.info("=" * 60)
    logger.info("开始加载数据集...")
    dataset = load_dataset(data_args)
    
    # 如果数据集需要预处理
    if "train" in dataset:
        train_dataset = dataset["train"]
    else:
        train_dataset = dataset
    
    logger.info(f"训练数据集大小: {len(train_dataset)}")
    logger.info("=" * 60)
    
    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        fp16=args.fp16,
        bf16=args.bf16,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=args.remove_unused_columns,
        report_to="tensorboard",
        logging_dir=f"{args.output_dir}/logs",
        # 显存优化选项
        gradient_checkpointing=True,  # 启用gradient checkpointing节省显存
        dataloader_pin_memory=False,  # 禁用pin_memory节省显存
        max_grad_norm=1.0,  # 梯度裁剪
    )
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # 开始训练
    logger.info("=" * 60)
    logger.info("准备开始训练...")
    logger.info(f"训练参数:")
    logger.info(f"  - 训练轮数: {args.num_train_epochs}")
    logger.info(f"  - Batch size: {args.per_device_train_batch_size}")
    logger.info(f"  - Gradient accumulation: {args.gradient_accumulation_steps}")
    logger.info(f"  - 有效batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
    logger.info(f"  - 总步数: ~{len(train_dataset) * args.num_train_epochs // (args.per_device_train_batch_size * args.gradient_accumulation_steps)}")
    logger.info("=" * 60)
    logger.info("开始训练... (GPU利用率会在训练循环开始后上升)")
    trainer.train()
    
    # 保存模型
    logger.info(f"保存模型到: {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info("训练完成！")


if __name__ == "__main__":
    main()

