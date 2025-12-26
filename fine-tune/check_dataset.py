#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""查看数据集信息"""

from datasets import load_from_disk
import os

dataset_path = "./data/dataset"

print("=" * 60)
print("数据集信息查看")
print("=" * 60)

try:
    # 加载数据集
    print(f"\n正在加载数据集: {dataset_path}")
    ds = load_from_disk(dataset_path)
    
    # 基本信息
    print(f"\n数据集大小: {len(ds):,} 条样本")
    print(f"\n特征结构:")
    print(ds.features)
    print(f"\n列名: {ds.column_names}")
    
    # 检查是否有split
    if hasattr(ds, 'keys'):
        print(f"\n数据集包含的split: {list(ds.keys())}")
        if 'train' in ds:
            train_ds = ds['train']
            print(f"训练集大小: {len(train_ds):,} 条")
            ds_to_check = train_ds
        else:
            ds_to_check = ds
    else:
        ds_to_check = ds
    
    # 查看前几条数据
    print(f"\n前3条数据示例:")
    print("-" * 60)
    for i in range(min(3, len(ds_to_check))):
        sample = ds_to_check[i]
        print(f"\n样本 {i+1}:")
        if 'input_ids' in sample:
            input_len = len(sample['input_ids'])
            print(f"  input_ids长度: {input_len}")
            print(f"  input_ids前30个token: {sample['input_ids'][:30]}")
        if 'labels' in sample:
            labels_len = len(sample['labels'])
            print(f"  labels长度: {labels_len}")
            print(f"  labels前30个token: {sample['labels'][:30]}")
        
        # 统计非-100的labels数量（-100通常表示忽略的token）
        if 'labels' in sample:
            non_ignore = sum(1 for x in sample['labels'] if x != -100)
            print(f"  有效labels数量: {non_ignore} (忽略: {labels_len - non_ignore})")
    
    # 统计信息
    print(f"\n统计信息:")
    print("-" * 60)
    if 'input_ids' in ds_to_check[0]:
        lengths = [len(ds_to_check[i]['input_ids']) for i in range(min(1000, len(ds_to_check)))]
        print(f"序列长度统计 (基于前1000条):")
        print(f"  平均长度: {sum(lengths) / len(lengths):.2f}")
        print(f"  最小长度: {min(lengths)}")
        print(f"  最大长度: {max(lengths)}")
    
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()

