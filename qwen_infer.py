#!/usr/bin/env python3
"""
Qwen3-4B Inference Script for UAR
This script is specifically configured for inference with Qwen3-4B model and UAR framework.
"""

import os
import argparse

def run_qwen_inference():
    """
    运行Qwen3-4B模型的UAR推理
    """
    
    parser = argparse.ArgumentParser(description="Qwen3-4B UAR Inference")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-4B", 
                       help="Qwen model name or path")
    parser.add_argument("--prompt_file", type=str, required=True,
                       help="Input prompt file path")
    parser.add_argument("--save_name", type=str, required=True, 
                       help="Output file path")
    parser.add_argument("--data_type", type=str, default="normal",
                       choices=["normal", "gsm8k", "drop"],
                       help="Data type for inference")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Inference batch size")
    
    args = parser.parse_args()
    
    # 构建推理命令
    cmd = f"""python uar_infer.py \\
    --model_name {args.model_name} \\
    --prompt_file {args.prompt_file} \\
    --save_name {args.save_name} \\
    --data_type {args.data_type} \\
    --batch_size {args.batch_size}"""
    
    print(f"运行Qwen3-4B UAR推理...")
    print(f"命令: {cmd}")
    
    # 执行命令
    os.system(cmd)

if __name__ == "__main__":
    run_qwen_inference()
