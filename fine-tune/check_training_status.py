#!/usr/bin/env python3
"""
检查训练状态脚本
"""
import os
from pathlib import Path

def check_training_status(output_dir="./output"):
    """检查训练状态"""
    output_path = Path(output_dir)
    
    print("=" * 60)
    print("训练状态检查")
    print("=" * 60)
    
    # 检查输出目录
    if not output_path.exists():
        print(f"❌ 输出目录不存在: {output_path}")
        return
    
    print(f"\n📁 输出目录: {output_path.absolute()}")
    
    # 检查模型文件
    model_files = []
    for pattern in ["*.safetensors", "*.bin", "adapter_config.json", "adapter_model.bin"]:
        model_files.extend(list(output_path.rglob(pattern)))
    
    if model_files:
        print(f"\n✅ 找到 {len(model_files)} 个模型文件:")
        for f in model_files[:10]:  # 只显示前10个
            size = f.stat().st_size / (1024 * 1024)  # MB
            print(f"  - {f.relative_to(output_path)} ({size:.2f} MB)")
    else:
        print("\n❌ 未找到模型文件（.safetensors, .bin, adapter_config.json）")
    
    # 检查checkpoint目录
    checkpoint_dirs = list(output_path.glob("checkpoint-*"))
    if checkpoint_dirs:
        print(f"\n✅ 找到 {len(checkpoint_dirs)} 个checkpoint目录:")
        for d in checkpoint_dirs[:5]:
            print(f"  - {d.name}")
    else:
        print("\n❌ 未找到checkpoint目录")
    
    # 检查日志文件
    log_files = list((output_path / "logs").glob("events.out.tfevents.*")) if (output_path / "logs").exists() else []
    if log_files:
        print(f"\n✅ 找到 {len(log_files)} 个TensorBoard日志文件")
        for f in log_files:
            size = f.stat().st_size / 1024  # KB
            print(f"  - {f.name} ({size:.2f} KB)")
    else:
        print("\n❌ 未找到TensorBoard日志文件")
    
    # 总结
    print("\n" + "=" * 60)
    if model_files or checkpoint_dirs:
        print("✅ 训练已完成，模型文件存在")
        print(f"\n模型路径: {output_path.absolute()}")
        print("\n可以在agent_framework中使用此模型:")
        print(f"  将模型复制到: agent_framework/lora/角色名/")
        print(f"  或修改角色配置中的LoRA路径指向: {output_path.absolute()}")
    else:
        print("❌ 训练未完成或模型未保存")
        print("\n可能的原因:")
        print("  1. 训练还在进行中")
        print("  2. 训练过程中出错，未保存模型")
        print("  3. 训练已完成但模型保存在其他位置")
        print("\n建议:")
        print("  1. 检查训练进程是否还在运行")
        print("  2. 查看训练日志确认训练状态")
        print("  3. 重新运行训练脚本")
    print("=" * 60)

if __name__ == "__main__":
    check_training_status()

