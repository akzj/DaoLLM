#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据处理脚本 - 将原始小说文本转换为训练用的Arrow格式数据集
"""

import os
import json
from pathlib import Path
from typing import List, Dict
from transformers import AutoTokenizer
from datasets import Dataset
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_novels(novels_dir: str) -> List[str]:
    """加载所有小说文本"""
    novels_dir = Path(novels_dir)
    novels = []
    
    txt_files = list(novels_dir.glob("*.txt"))
    logger.info(f"找到 {len(txt_files)} 个小说文件")
    
    for txt_file in txt_files:
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    novels.append(content)
                    logger.info(f"已加载: {txt_file.name} ({len(content)} 字符)")
        except Exception as e:
            logger.warning(f"加载 {txt_file.name} 失败: {e}")
    
    logger.info(f"总共加载 {len(novels)} 本小说，总字符数: {sum(len(n) for n in novels):,}")
    return novels


def split_text_into_chunks(text: str, chunk_size: int = 50000) -> List[str]:
    """将长文本按字符数切分成较小的块"""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks


def tokenize_texts(
    texts: List[str],
    tokenizer: AutoTokenizer,
    max_length: int = 2048,
    stride: int = 512
) -> Dict[str, List]:
    """将文本tokenize并切分成固定长度的序列"""
    input_ids_list = []
    labels_list = []
    
    logger.info(f"开始tokenize，最大长度: {max_length}, 步长: {stride}")
    
    # 估算每个chunk的字符数（保守估计，假设平均每个token对应2-3个中文字符）
    chunk_char_size = max_length * 2  # 保守估计
    
    for text_idx, text in enumerate(texts):
        if (text_idx + 1) % 10 == 0:
            logger.info(f"处理进度: {text_idx + 1}/{len(texts)}")
        
        # 如果文本太长，先按字符切分成较小的块
        if len(text) > chunk_char_size:
            text_chunks = split_text_into_chunks(text, chunk_char_size)
        else:
            text_chunks = [text]
        
        # 对每个文本块进行处理
        for chunk_text in text_chunks:
            # Tokenize文本块（添加truncation防止过长）
            tokens = tokenizer(
                chunk_text,
                add_special_tokens=False,
                return_attention_mask=False,
                truncation=True,
                max_length=max_length * 10,  # 允许较大的token序列，后续会切分
            )['input_ids']
            
            # 使用滑动窗口切分
            for i in range(0, len(tokens), stride):
                chunk = tokens[i:i + max_length]
                
                # 如果chunk太短，跳过
                if len(chunk) < max_length // 2:
                    continue
                
                # 填充到max_length
                if len(chunk) < max_length:
                    chunk = chunk + [tokenizer.pad_token_id] * (max_length - len(chunk))
                
                input_ids_list.append(chunk)
                # 对于causal LM，labels和input_ids相同（除了padding部分）
                # 将padding部分的label设为-100（忽略）
                labels = chunk.copy()
                for j in range(len(labels)):
                    if labels[j] == tokenizer.pad_token_id:
                        labels[j] = -100
                labels_list.append(labels)
    
    logger.info(f"生成了 {len(input_ids_list)} 个训练样本")
    return {
        'input_ids': input_ids_list,
        'labels': labels_list
    }


def create_dataset(
    novels_dir: str,
    output_dir: str,
    model_name: str = "Qwen/Qwen1.5-7B-Chat",
    max_length: int = 2048,
    stride: int = 512
):
    """创建数据集"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("开始处理数据")
    logger.info(f"模型: {model_name}")
    logger.info(f"输入目录: {novels_dir}")
    logger.info(f"输出目录: {output_dir}")
    logger.info("=" * 60)
    
    # 加载tokenizer
    logger.info(f"加载tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 加载小说文本
    novels = load_novels(novels_dir)
    
    if not novels:
        raise ValueError("没有找到任何小说文本！")
    
    # Tokenize和切分
    data = tokenize_texts(novels, tokenizer, max_length, stride)
    
    # 创建Dataset
    logger.info("创建Dataset对象...")
    dataset = Dataset.from_dict(data)
    
    # 保存数据集
    logger.info(f"保存数据集到: {output_path}")
    dataset.save_to_disk(str(output_path))
    
    # 保存tokenizer（可选）
    tokenizer.save_pretrained(str(output_path))
    
    logger.info("=" * 60)
    logger.info("数据处理完成！")
    logger.info(f"数据集大小: {len(dataset):,} 个样本")
    logger.info(f"保存位置: {output_path}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="准备训练数据")
    parser.add_argument(
        "--novels_dir",
        type=str,
        default="./data/novels_raw",
        help="原始小说文本目录"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/dataset",
        help="输出数据集目录"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen1.5-7B-Chat",
        help="模型名称（用于tokenizer）"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="最大序列长度"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=512,
        help="滑动窗口步长"
    )
    
    args = parser.parse_args()
    
    create_dataset(
        novels_dir=args.novels_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        max_length=args.max_length,
        stride=args.stride
    )


if __name__ == "__main__":
    main()

