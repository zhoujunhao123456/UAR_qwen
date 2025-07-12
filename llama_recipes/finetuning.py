# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire
import random
import torch
import torch.optim as optim

from torch.optim.lr_scheduler import StepLR
from transformers import (
    LlamaForSequenceClassification,
    LlamaTokenizer,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from llama_recipes.configs import train_config as TRAIN_CONFIG
from llama_recipes.data.concatenator import ConcatDataset

from llama_recipes.utils.config_utils import (
    update_config,
    generate_dataset_config,
    get_dataloader_kwargs,
)
from llama_recipes.utils.dataset_utils import get_preprocessed_dataset

from llama_recipes.utils.train_utils import (
    train_reward_model,
    print_model_size,
)

def main(**kwargs):
    # Update the configuration for the training and sharding process
    train_config = TRAIN_CONFIG()
    update_config((train_config), **kwargs)

    # Set the seeds for reproducibility
    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)

    # Load the tokenizer and add special tokens
    model_name_lower = train_config.model_name.lower()
    is_qwen_model = "qwen" in model_name_lower or "Qwen" in train_config.model_name
    
    if is_qwen_model:
        tokenizer = AutoTokenizer.from_pretrained(train_config.model_name, trust_remote_code=True)
        # 确保Qwen tokenizer有正确的pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        tokenizer = LlamaTokenizer.from_pretrained(train_config.model_name)
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load the pre-trained model and setup its configuration
    use_cache = False if train_config.enable_fsdp else None

    if is_qwen_model:
        model = AutoModelForSequenceClassification.from_pretrained(
            train_config.model_name,
            load_in_8bit=True if train_config.quantization else None,
            device_map="auto" if train_config.quantization else None,
            use_cache=use_cache,
            num_labels=1 if train_config.reward_model_loss_type == "mse" else 2,
            trust_remote_code=True,
            # 为Qwen模型添加pad_token_id配置
            pad_token_id=tokenizer.pad_token_id,
        )
    else:
        model = LlamaForSequenceClassification.from_pretrained(
            train_config.model_name,
            load_in_8bit=True if train_config.quantization else None,
            device_map="auto" if train_config.quantization else None,
            use_cache=use_cache,
            num_labels=1 if train_config.reward_model_loss_type == "mse" else 2,
        )

    # 确保模型和tokenizer的pad_token_id一致
    if is_qwen_model:
        # 对于Qwen模型，需要额外确保配置正确
        model.config.pad_token_id = tokenizer.pad_token_id
        
        # 安全检查generation_config
        if hasattr(model, 'generation_config') and model.generation_config is not None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id
        
        # 强制设置模型的pad_token_id，确保训练时正确识别
        print(f"Qwen模型配置:")
        print(f"  tokenizer.pad_token: {tokenizer.pad_token}")
        print(f"  tokenizer.pad_token_id: {tokenizer.pad_token_id}")
        print(f"  model.config.pad_token_id: {model.config.pad_token_id}")
        print(f"  model.generation_config: {model.generation_config}")
        
        # 确保所有相关配置都设置正确
        if hasattr(model.config, 'pad_token_id'):
            model.config.pad_token_id = tokenizer.pad_token_id
        
        # 验证pad_token设置
        if model.config.pad_token_id is None:
            print("警告: 模型pad_token_id仍为None，强制设置为eos_token_id")
            model.config.pad_token_id = tokenizer.eos_token_id
    
    model.pad_token_id = tokenizer.pad_token_id

    if train_config.only_cls_for_rmce:
        for param in model.model.parameters():
            param.requires_grad = False
        for param in model.score.parameters():
            param.requires_grad = True
        
    print_model_size(model, train_config, 0)

    model.to("cuda")

    dataset_config = generate_dataset_config(train_config, kwargs)

     # Load and preprocess the dataset for training and validation
    dataset_train = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="train",
    )

    dataset_val = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="test",
    )

    print(f"--> Validation Set Length = {len(dataset_val)}")

    if train_config.batching_strategy == "packing":
        dataset_train = ConcatDataset(dataset_train, chunk_size=train_config.context_length)

    train_dl_kwargs = get_dataloader_kwargs(train_config, dataset_train, tokenizer, "train")

    # Create DataLoaders for the training and validation dataset
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        **train_dl_kwargs,
    )

    eval_dataloader = None
    if train_config.run_validation:
        if train_config.batching_strategy == "packing":
            dataset_val = ConcatDataset(dataset_val, chunk_size=train_config.context_length)

        val_dl_kwargs = get_dataloader_kwargs(train_config, dataset_val, tokenizer, "val")

        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            **val_dl_kwargs,
        )

    # Initialize the optimizer and learning rate scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)

    # Create a minimal fsdp_config for compatibility
    from dataclasses import dataclass
    from torch.distributed.fsdp import StateDictType
    
    @dataclass
    class MinimalFSDPConfig:
        checkpoint_type: StateDictType = StateDictType.FULL_STATE_DICT
        
    fsdp_config = MinimalFSDPConfig()

    # Start the training process
    results = train_reward_model(
        model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        scheduler,
        train_config.gradient_accumulation_steps,
        train_config,
        fsdp_config=fsdp_config,
        local_rank=None,
        rank=0
    )
    
    [print(f'Key: {k}, Value: {v}') for k, v in results.items()]

if __name__ == "__main__":
    fire.Fire(main)
