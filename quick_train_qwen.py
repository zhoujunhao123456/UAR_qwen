#!/usr/bin/env python3
"""
Qwen3-4B Training Script for UAR - Storage Optimized Version
This script only saves UAR-specific files to minimize storage usage.
"""

import os
import sys
import shutil
from llama_recipes.finetuning import main

def cleanup_large_model_files(output_dir):
    """
    清理大型模型文件，只保留UAR需要的核心文件
    """
    # UAR实际需要的文件
    keep_files = {
        "config.json",           # 模型配置
        "tokenizer_config.json", # 分词器配置
        "tokenizer.json",        # 分词器文件
        "special_tokens_map.json", # 特殊token映射
        "filtered_model.pth"     # UAR分类头权重（核心文件）
    }
    
    # 需要删除的大型文件
    large_files_to_remove = [
        "pytorch_model.bin",
        "model.safetensors", 
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        # 可能的其他分片文件
    ]
    
    removed_files = []
    saved_space = 0
    
    if not os.path.exists(output_dir):
        return removed_files, saved_space
    
    print(f"🧹 清理目录: {output_dir}")
    
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        
        # 如果是大型模型文件且不在保留列表中
        if filename in large_files_to_remove or (
            filename.startswith("model-") and filename.endswith(".safetensors")
        ):
            try:
                file_size = os.path.getsize(file_path)
                os.remove(file_path)
                removed_files.append(filename)
                saved_space += file_size
                print(f"  ✓ 删除: {filename} ({file_size / (1024**3):.2f}GB)")
            except Exception as e:
                print(f"  ✗ 删除失败: {filename} - {e}")
    
    return removed_files, saved_space

def verify_uar_files(output_dir):
    """
    验证UAR核心文件是否存在
    """
    required_files = [
        "config.json",
        "filtered_model.pth"  # UAR分类头权重
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(output_dir, file)):
            missing_files.append(file)
    
    return len(missing_files) == 0, missing_files

def train_qwen_classifiers_optimized(model_path="Qwen/Qwen2.5-4B", output_base_dir="./qwen_classifiers", 
                                   skip_completed=True, cleanup_after_training=True):
    """
    Train UAR classifiers with storage optimization
    """
    
    print("="*60)
    print("UAR Qwen3-4B 分类器训练系统 - 存储优化版本")
    print("目标：只保存UAR核心文件，节省存储空间")
    print("="*60)
    
    # 模型路径检测（复用原有逻辑）
    current_dir = os.getcwd()
    print(f"当前工作目录: {current_dir}")
    
    local_model_paths = [
        "/root/autodl-tmp/models/Qwen3-4B",
        "/root/autodl-tmp/models/Qwen2.5-4B",
        "/root/models/Qwen3-4B", 
        "/root/models/Qwen2.5-4B",
        "./models/Qwen3-4B",
        "./models/Qwen2.5-4B",
        "/root/autodl-tmp/UAR/models/Qwen3-4B",
        "/root/autodl-tmp/UAR/models/Qwen2.5-4B",
        model_path
    ]
    
    actual_model_path = None
    for path in local_model_paths:
        if os.path.exists(path) and os.path.isdir(path):
            model_files = ["config.json", "pytorch_model.bin", "model.safetensors"]
            has_model_files = any(os.path.exists(os.path.join(path, f)) for f in model_files)
            if has_model_files:
                actual_model_path = path
                print(f"✓ 找到本地模型: {path}")
                break
    
    if actual_model_path is None:
        print(f"❌ 未找到本地模型")
        return False
    
    model_path = actual_model_path
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 训练参数（复用原有配置）
    common_args = {
        "model_name": model_path,
        "batch_size_training": 32,
        "val_batch_size": 32,
        "batching_strategy": "padding",
        "context_length": 4096,
        "gradient_accumulation_steps": 1,
        "lr": 5e-5,
        "weight_decay": 0.0,
        "gamma": 0.85,
        "num_epochs": 3,
        "num_workers_dataloader": 4,
        "reward_model_loss_type": "ce",
        "only_cls_for_rmce": True,  # UAR核心：只训练分类头
        "use_fp16": True,
        "mixed_precision": True,
        "run_validation": True,
        "save_model": True,
        "quantization": False,
        "use_peft": False,
        "enable_fsdp": False,
        "one_gpu": True,
        "save_optimizer": False,
        "seed": 42,
    }
    
    # 分类器配置
    classifiers = [
        {
            "name": "time_aware",
            "dataset": "time_aware_cls_ce",
            "output_dir": f"{output_base_dir}/Time_aware_qwen3_4b",
            "description": "Time Awareness Classifier"
        },
        {
            "name": "knowledge_aware", 
            "dataset": "knowledge_aware_cls_ce",
            "output_dir": f"{output_base_dir}/Know_aware_qwen3_4b",
            "description": "Knowledge Awareness Classifier"
        },
        {
            "name": "self_aware",
            "dataset": "self_aware_cls_ce_llama2_7b_chat",
            "output_dir": f"{output_base_dir}/Self_aware_qwen3_4b",
            "description": "Self Awareness Classifier"
        },
        {
            "name": "intent_aware",
            "dataset": "intent_aware_cls_ce", 
            "output_dir": f"{output_base_dir}/Intent_aware_qwen3_4b",
            "description": "Intent Awareness Classifier"
        }
    ]
    
    print(f"存储优化设置:")
    print(f"  - 训练后自动清理: {cleanup_after_training}")
    print(f"  - 只保留UAR核心文件: config.json, tokenizer文件, filtered_model.pth")
    print(f"  - 删除大型模型文件: pytorch_model.bin, model.safetensors等")
    print("="*60)
    
    successful_classifiers = []
    failed_classifiers = []
    skipped_classifiers = []
    total_saved_space = 0
    
    for i, classifier in enumerate(classifiers, 1):
        output_dir = classifier['output_dir']
        
        # 检查是否已训练完成
        if skip_completed:
            is_complete, missing = verify_uar_files(output_dir)
            if is_complete:
                print(f"\n[{i}/{len(classifiers)}] 跳过 {classifier['name']} - 已训练完成")
                successful_classifiers.append(classifier['name'])
                skipped_classifiers.append(classifier['name'])
                continue
        
        print(f"\n[{i}/{len(classifiers)}] 训练 {classifier['description']} ({classifier['name']})...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        train_args = {
            **common_args,
            "dataset": classifier["dataset"],
            "output_dir": output_dir
        }
        
        try:
            print(f"开始训练...")
            result = main(**train_args)
            print(f"✓ {classifier['name']} 训练完成")
            
            # 训练完成后立即清理大型文件
            if cleanup_after_training:
                print(f"🧹 开始清理存储...")
                removed_files, saved_space = cleanup_large_model_files(output_dir)
                total_saved_space += saved_space
                
                if removed_files:
                    print(f"✓ 已删除 {len(removed_files)} 个大型文件")
                    print(f"✓ 节省空间: {saved_space / (1024**3):.2f}GB")
                else:
                    print(f"ℹ️  未找到需要清理的大型文件")
            
            # 验证UAR文件完整性
            is_complete, missing = verify_uar_files(output_dir)
            if is_complete:
                print(f"✓ UAR核心文件验证通过")
                successful_classifiers.append(classifier['name'])
            else:
                print(f"⚠️  UAR文件不完整，缺少: {missing}")
                
        except Exception as e:
            print(f"✗ {classifier['name']} 训练失败: {e}")
            failed_classifiers.append(classifier['name'])
            continue
    
    # 训练完成总结
    print("\n" + "="*60)
    print("训练完成总结")
    print("="*60)
    print(f"成功训练: {len(successful_classifiers)}/{len(classifiers)} 个分类器")
    
    if successful_classifiers:
        print(f"✓ 成功: {', '.join(successful_classifiers)}")
    
    if skipped_classifiers:
        print(f"⏭️  跳过: {', '.join(skipped_classifiers)}")
        
    if failed_classifiers:
        print(f"✗ 失败: {', '.join(failed_classifiers)}")
    
    if total_saved_space > 0:
        print(f"💾 总节省空间: {total_saved_space / (1024**3):.2f}GB")
    
    print("="*60)
    
    return len(successful_classifiers) == len(classifiers)

def cleanup_existing_classifiers(base_dir="./qwen_classifiers"):
    """
    清理已有分类器目录中的大型文件
    """
    print("🧹 清理现有分类器目录...")
    
    if not os.path.exists(base_dir):
        print(f"目录不存在: {base_dir}")
        return
    
    total_saved = 0
    cleaned_dirs = []
    
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            # 先验证UAR文件是否完整
            is_complete, missing = verify_uar_files(item_path)
            if is_complete:
                removed_files, saved_space = cleanup_large_model_files(item_path)
                if saved_space > 0:
                    total_saved += saved_space
                    cleaned_dirs.append(item)
                    print(f"✓ {item}: 节省 {saved_space / (1024**3):.2f}GB")
            else:
                print(f"⚠️  跳过 {item}: UAR文件不完整 {missing}")
    
    if cleaned_dirs:
        print(f"\n🎉 清理完成!")
        print(f"清理目录: {len(cleaned_dirs)} 个")
        print(f"总节省空间: {total_saved / (1024**3):.2f}GB")
    else:
        print(f"ℹ️  未找到可清理的目录")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="UAR Training with Storage Optimization")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-4B")
    parser.add_argument("--output_dir", type=str, default="./qwen_classifiers")
    parser.add_argument("--skip_completed", action="store_true", default=True)
    parser.add_argument("--cleanup", action="store_true", default=True,
                       help="Cleanup large model files after training")
    parser.add_argument("--cleanup_only", action="store_true", default=False,
                       help="Only cleanup existing directories, don't train")
    
    args = parser.parse_args()
    
    if args.cleanup_only:
        cleanup_existing_classifiers(args.output_dir)
    else:
        train_qwen_classifiers_optimized(
            model_path=args.model_path,
            output_base_dir=args.output_dir,
            skip_completed=args.skip_completed,
            cleanup_after_training=args.cleanup
        )
