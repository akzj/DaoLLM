#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理脚本 - 用于测试微调后的Qwen1.5-7B-Chat模型
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse


def load_model(base_model_path, lora_model_path, use_4bit=True):
    """加载基础模型和LoRA权重"""
    print(f"加载基础模型: {base_model_path}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        lora_model_path if lora_model_path else base_model_path,
        trust_remote_code=True,
    )
    
    # 配置量化（如果使用）
    if use_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None
    
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if not use_4bit else None,
    )
    
    # 如果有LoRA权重，加载LoRA
    if lora_model_path:
        print(f"加载LoRA权重: {lora_model_path}")
        model = PeftModel.from_pretrained(base_model, lora_model_path)
        # 合并权重（可选，用于更快推理）
        # model = model.merge_and_unload()
    else:
        model = base_model
    
    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_length=512, temperature=0.7, top_p=0.9):
    """生成回复"""
    # 格式化prompt（Qwen1.5-Chat格式）
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # 应用chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # 解码
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response


def main():
    parser = argparse.ArgumentParser(description="推理脚本")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen1.5-7B-Chat",
                       help="基础模型路径")
    parser.add_argument("--lora_model", type=str, default="./output",
                       help="LoRA模型路径")
    parser.add_argument("--use_4bit", action="store_true", default=True,
                       help="使用4bit量化")
    parser.add_argument("--prompt", type=str, default="你好，请介绍一下修仙世界",
                       help="输入提示")
    parser.add_argument("--max_length", type=int, default=512,
                       help="最大生成长度")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="温度参数")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="top_p参数")
    parser.add_argument("--interactive", action="store_true",
                       help="交互模式")
    
    args = parser.parse_args()
    
    # 加载模型
    model, tokenizer = load_model(args.base_model, args.lora_model, args.use_4bit)
    
    if args.interactive:
        # 交互模式
        print("=" * 50)
        print("进入交互模式，输入 'quit' 或 'exit' 退出")
        print("=" * 50)
        
        while True:
            try:
                prompt = input("\n你: ")
                if prompt.lower() in ['quit', 'exit', '退出']:
                    break
                
                if not prompt.strip():
                    continue
                
                print("\nAI: ", end="", flush=True)
                response = generate_response(
                    model, tokenizer, prompt,
                    max_length=args.max_length,
                    temperature=args.temperature,
                    top_p=args.top_p
                )
                print(response)
                
            except KeyboardInterrupt:
                print("\n\n退出交互模式")
                break
            except Exception as e:
                print(f"\n错误: {e}")
    else:
        # 单次推理
        print(f"输入: {args.prompt}")
        print("\n输出:")
        response = generate_response(
            model, tokenizer, args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p
        )
        print(response)


if __name__ == "__main__":
    main()

