# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class train_config:
    model_name: str="Qwen/Qwen3-4B" # 默认使用Qwen模型，可通过参数覆盖
    enable_fsdp: bool=False
    low_cpu_fsdp: bool=False
    run_validation: bool=True
    batch_size_training: int=4
    batching_strategy: str="packing" #alternative: padding
    # batching_strategy: str="padding"
    context_length: int=4096
    gradient_accumulation_steps: int=1
    num_epochs: int=10
    num_workers_dataloader: int=1
    lr: float=1e-4
    weight_decay: float=0.0
    gamma: float= 0.85
    seed: int=42
    use_fp16: bool=False
    mixed_precision: bool=True
    val_batch_size: int=1
    dataset = "samsum_dataset"
    peft_method: str = "lora" # None , llama_adapter, prefix
    use_peft: bool=False
    output_dir: str = "PATH/to/save/PEFT/model"
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    quantization: bool = False
    one_gpu: bool = False
    save_model: bool = True
    dist_checkpoint_root_folder: str="PATH/to/save/FSDP/model" # will be used if using FSDP
    dist_checkpoint_folder: str="fine-tuned" # will be used if using FSDP
    save_optimizer: bool=False # will be used if using FSDP
    use_fast_kernels: bool = False # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    train_reward_model: bool = False
    train_dpo: bool = False
    reward_model_loss_type: str = "mse" # alternative: "ce", "bce"
    train_ppo_reward_model: bool = False
    train_ppo_policy_model: bool = False
    reward_model_path: str = None
    train_ppo_reward_model_trl: bool = False
    train_ppo_policy_model_trl: bool = False
    only_cls_for_rmce: bool = False