#!/usr/bin/env python3
"""
安全版Qwen训练脚本
包含详细的错误检查和调试信息
"""

import os
import sys

def check_environment():
    """检查环境和依赖"""
    print("=== 环境检查 ===")
    
    # 检查GPU
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU数量: {torch.cuda.device_count()}")
            print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"GPU检查失败: {e}")
        return False
    
    # 检查训练数据
    print(f"\n训练数据检查:")
    if not os.path.exists("training_data"):
        print("✗ training_data目录不存在")
        return False
    
    required_files = [
        "training_data/time_aware_cls_ce_train.json",
        "training_data/time_aware_cls_ce_valid.json",
        "training_data/knowledge_aware_cls_ce_train.json", 
        "training_data/knowledge_aware_cls_ce_valid.json",
        "training_data/self_aware_cls_ce_llama2_7b_chat_train.json",
        "training_data/self_aware_cls_ce_llama2_7b_chat_valid.json",
        "training_data/intent_aware_cls_ce_train.json",
        "training_data/intent_aware_cls_ce_valid.json"
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n缺少 {len(missing_files)} 个训练数据文件")
        return False
    
    return True

def train_single_classifier(model_path, dataset_name, output_dir, batch_size=15, epochs=1):
    """训练单个分类器"""
    print(f"\n=== 训练 {dataset_name} 分类器 ===")
    
    try:
        from llama_recipes.finetuning import main
        
        train_args = {
            "model_name": model_path,
            "output_dir": output_dir,
            "dataset": dataset_name,
            "batch_size_training": batch_size,
            "batching_strategy": "padding",
            "lr": 3e-5,
            "num_epochs": epochs,
            "reward_model_loss_type": "ce",
            "only_cls_for_rmce": True,
            "use_fp16": True,
        }
        
        print(f"训练参数:")
        for key, value in train_args.items():
            print(f"  {key}: {value}")
        
        # 测试tokenizer配置
        print(f"\n测试tokenizer配置:")
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            print(f"  pad_token: {tokenizer.pad_token}")
            print(f"  pad_token_id: {tokenizer.pad_token_id}")
            print(f"  eos_token: {tokenizer.eos_token}")
            print(f"  eos_token_id: {tokenizer.eos_token_id}")
            
            # 如果pad_token为None，设置它
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
                print(f"  设置pad_token为eos_token: {tokenizer.pad_token}")
        except Exception as e:
            print(f"  tokenizer测试失败: {e}")
        
        # 调用训练函数
        main(**train_args)
        print(f"✓ {dataset_name} 训练成功")
        return True
        
    except Exception as e:
        print(f"✗ {dataset_name} 训练失败: {e}")
        print(f"错误类型: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("====================================")
    print("安全版Qwen3-4B UAR训练脚本")
    print("====================================")
    
    # 环境检查
    if not check_environment():
        print("\n环境检查失败，请解决上述问题后重试")
        return
    
    # 确定模型路径
    if os.path.exists("models/Qwen3-4B"):
        model_path = "models/Qwen3-4B"
        print(f"\n使用本地模型: {model_path}")
    else:
        model_path = "Qwen/Qwen2.5-4B"
        print(f"\n使用HuggingFace模型: {model_path}")
    
    # 创建输出目录
    output_base = "./qwen_classifiers"
    os.makedirs(output_base, exist_ok=True)
    
    
    # 训练分类器列表
    classifiers = [
        ("time_aware_cls_ce", "time_aware"),
        ("knowledge_aware_cls_ce", "knowledge_aware"), 
        ("self_aware_cls_ce_llama2_7b_chat", "self_aware"),
        ("intent_aware_cls_ce", "intent_aware")
    ]
    
    print(f"\n=== 开始训练分类器 ===")
    
    success_count = 0
    for dataset_name, output_name in classifiers:
        output_dir = f"{output_base}/{output_name}_qwen3_4b"
        
        if train_single_classifier(model_path, dataset_name, output_dir, batch_size=1, epochs=1):
            success_count += 1
    
    print(f"\n====================================")
    print(f"训练完成！")
    print(f"成功: {success_count}/{len(classifiers)} 个分类器")
    print(f"输出目录: {output_base}")
    print(f"====================================")

if __name__ == "__main__":
    main()
