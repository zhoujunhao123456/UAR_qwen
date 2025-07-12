# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import copy
import json

import torch
from torch.utils.data import Dataset

B_INST, E_INST = "[INST]", "[/INST]"

def detect_model_type(tokenizer):
    """检测模型类型以选择合适的提示格式"""
    tokenizer_class = tokenizer.__class__.__name__
    if "Qwen" in tokenizer_class:
        return "qwen"
    elif "Llama" in tokenizer_class:
        return "llama"
    else:
        return "unknown"

def format_prompt(question, model_type="llama"):
    """根据模型类型格式化提示"""
    if model_type == "qwen":
        # Qwen模型的对话格式
        return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    else:
        # LLaMA模型的对话格式
        return f"{B_INST} {question} {E_INST}"

class RewardDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train"):
        if partition == 'train':
            self.ann = json.load(open(dataset_config.train_data_path))
        else:
            self.ann = json.load(open(dataset_config.valid_data_path))
        self.tokenizer = tokenizer
        self.model_type = detect_model_type(tokenizer)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]
        example = format_prompt(ann['question'], self.model_type)
        
        # 使用tokenizer的call方法而不是encode，以确保正确处理
        try:
            tokenized = self.tokenizer(
                example,
                truncation=True,
                padding=False,
                return_tensors="pt"
            )
            example = tokenized["input_ids"].squeeze(0)
            example_mask = tokenized["attention_mask"].squeeze(0)
        except Exception as e:
            # 如果tokenizer调用失败，尝试用encode方法
            print(f"Tokenizer call failed: {e}, trying encode method")
            example = self.tokenizer.encode(example)
            example = torch.tensor(example, dtype=torch.int64)
            example_mask = example.ge(0)
        
        example[~example_mask] = 0

        return {
            "input_ids": example.tolist(),
            "labels": [ann['label']],
            "attention_mask": example_mask.tolist(),
        }
