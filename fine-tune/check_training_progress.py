#!/usr/bin/env python3
"""
检查训练进度脚本
"""
import os
import time
from pathlib import Path
import subprocess

def check_training_progress():
    """检查训练进度"""
    print("=" * 60)
    print("训练进度检查")
    print("=" * 60)
    
    # 检查进程
    result = subprocess.run(
        ["ps", "aux"], 
        capture_output=True, 
        text=True
    )
    
    train_processes = [line for line in result.stdout.split('\n') if 'train_lora' in line and 'grep' not in line]
    
    if train_processes:
        print("\n✅ 训练进程正在运行:")
        for proc in train_processes:
            print(f"  {proc}")
    else:
        print("\n❌ 未找到训练进程")
        return
    
    # 检查GPU使用
    print("\n📊 GPU状态:")
    result = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader"], 
                          capture_output=True, text=True)
    print(result.stdout)
    
    # 检查日志文件
    log_dir = Path("output/logs")
    if log_dir.exists():
        log_files = list(log_dir.glob("events.out.tfevents.*"))
        if log_files:
            latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
            size = latest_log.stat().st_size / 1024  # KB
            mtime = time.ctime(latest_log.stat().st_mtime)
            print(f"\n📝 最新日志文件: {latest_log.name}")
            print(f"   大小: {size:.2f} KB")
            print(f"   修改时间: {mtime}")
            
            # 尝试读取TensorBoard日志
            try:
                from tensorboard.backend.event_processing import event_accumulator
                ea = event_accumulator.EventAccumulator(str(latest_log))
                ea.Reload()
                tags = ea.Tags()
                
                if 'scalars' in tags:
                    scalars = tags['scalars']
                    print(f"\n📈 训练指标:")
                    for tag in scalars[:5]:  # 只显示前5个
                        values = ea.Scalars(tag)
                        if values:
                            latest = values[-1]
                            print(f"   {tag}: {latest.value:.4f} (步数: {latest.step})")
                
                if 'train/loss' in scalars:
                    losses = ea.Scalars('train/loss')
                    print(f"\n🎯 训练进度:")
                    print(f"   总步数: {len(losses)}")
                    if losses:
                        print(f"   最新loss: {losses[-1].value:.4f}")
                        print(f"   初始loss: {losses[0].value:.4f}")
            except Exception as e:
                print(f"\n⚠️  无法读取TensorBoard日志: {e}")
    
    # 检查模型文件
    output_dir = Path("output")
    model_files = list(output_dir.glob("**/*.safetensors")) + list(output_dir.glob("**/*.bin"))
    if model_files:
        print(f"\n✅ 找到 {len(model_files)} 个模型文件")
    else:
        print("\n⏳ 尚未保存模型文件（训练可能还在进行中）")
    
    print("\n" + "=" * 60)
    print("说明:")
    print("- GPU利用率0%可能是正常的，如果正在加载/预处理数据")
    print("- 数据加载阶段通常是CPU密集型，GPU利用率会较低")
    print("- 训练循环开始后，GPU利用率应该会上升")
    print("- 如果长时间（>5分钟）GPU利用率仍为0%，可能需要检查")
    print("=" * 60)

if __name__ == "__main__":
    check_training_progress()

