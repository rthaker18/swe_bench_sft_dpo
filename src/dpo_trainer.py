"""
Direct Preference Optimization (DPO) trainer for SWE-Bench patch generation.

DPO aligns the model to prefer correct patches over incorrect ones
without needing a separate reward model.
"""

import os
from typing import Dict, Optional
from dataclasses import dataclass

import torch
import yaml
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from trl import DPOTrainer, DPOConfig

import wandb


@dataclass
class DPOTrainingConfig:
    """Configuration for DPO training."""
    # Model (start from SFT checkpoint)
    model_name: str = "./outputs/sft"
    torch_dtype: str = "bfloat16"
    
    # Quantization
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # DPO specific
    beta: float = 0.1  # KL penalty
    loss_type: str = "sigmoid"
    
    # Training
    output_dir: str = "./outputs/dpo"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-7  # Much lower for DPO
    max_length: int = 4096
    max_prompt_length: int = 2048
    
    # Hub
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    
    @classmethod
    def from_yaml(cls, path: str) -> "DPOTrainingConfig":
        """Load config from YAML file."""
        with open(path) as f:
            config_dict = yaml.safe_load(f)
        
        flat_config = {}
        for section, values in config_dict.items():
            if isinstance(values, dict):
                for k, v in values.items():
                    flat_config[k] = v
            else:
                flat_config[section] = values
        
        return cls(**{k: v for k, v in flat_config.items() 
                     if k in cls.__dataclass_fields__})


def get_bnb_config(config: DPOTrainingConfig) -> BitsAndBytesConfig:
    """Create BitsAndBytes quantization config."""
    compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)
    
    return BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=True,
    )


def get_lora_config(config: DPOTrainingConfig) -> LoraConfig:
    """Create LoRA configuration."""
    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_model_for_dpo(config: DPOTrainingConfig):
    """
    Load model for DPO training.
    
    If starting from a PEFT checkpoint, load the base + adapter.
    Then add a new trainable adapter for DPO.
    """
    print(f"Loading model: {config.model_name}")
    
    # Check if this is a PEFT checkpoint
    is_peft = os.path.exists(os.path.join(config.model_name, "adapter_config.json"))
    
    # Quantization config
    bnb_config = get_bnb_config(config)
    
    if is_peft:
        # Load the base model from the adapter config
        from peft import PeftConfig
        peft_config = PeftConfig.from_pretrained(config.model_name)
        base_model_name = peft_config.base_model_name_or_path
        
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=getattr(torch, config.torch_dtype),
        )
        
        # Load the SFT adapter
        model = PeftModel.from_pretrained(model, config.model_name)
        
        # Merge and unload to get a clean model for new adapters
        model = model.merge_and_unload()
        
        # Prepare for training again
        model = prepare_model_for_kbit_training(model)
        
        # Add new LoRA adapter for DPO
        lora_config = get_lora_config(config)
        model = get_peft_model(model, lora_config)
        
        # Tokenizer from original
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True,
        )
    else:
        # Load directly (not a PEFT checkpoint)
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=getattr(torch, config.torch_dtype),
        )
        
        model = prepare_model_for_kbit_training(model)
        
        lora_config = get_lora_config(config)
        model = get_peft_model(model, lora_config)
        
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True,
        )
    
    # Set padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model.print_trainable_parameters()
    
    return model, tokenizer


def prepare_preference_dataset(
    dataset_path: str,
) -> Dataset:
    """
    Load preference dataset for DPO.
    
    Expected format:
    {
        "prompt": "...",
        "chosen": "correct response",
        "rejected": "incorrect response"
    }
    """
    if os.path.exists(dataset_path):
        if dataset_path.endswith(".json"):
            dataset = Dataset.from_json(dataset_path)
        else:
            dataset = Dataset.from_parquet(dataset_path)
    else:
        # Try loading from HuggingFace
        dataset = load_dataset(dataset_path, split="train")
    
    print(f"Loaded {len(dataset)} preference pairs")
    
    # Validate format
    required_fields = ["prompt", "chosen", "rejected"]
    for field in required_fields:
        if field not in dataset.column_names:
            raise ValueError(f"Dataset missing required field: {field}")
    
    return dataset


def create_dpo_trainer(
    model,
    tokenizer,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset],
    config: DPOTrainingConfig,
) -> DPOTrainer:
    """Create DPO trainer instance."""
    
    training_args = DPOConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        optim="paged_adamw_32bit",
        fp16=False,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=50,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=50 if eval_dataset else None,
        gradient_checkpointing=True,
        max_length=config.max_length,
        max_prompt_length=config.max_prompt_length,
        beta=config.beta,
        loss_type=config.loss_type,
        push_to_hub=config.push_to_hub,
        hub_model_id=config.hub_model_id,
        report_to=["wandb"] if wandb.run else ["tensorboard"],
        remove_unused_columns=False,
    )
    
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    
    return trainer


def train_dpo(
    config_path: str = None,
    config: DPOTrainingConfig = None,
    dataset_path: str = "./data/preference_pairs.json",
    wandb_project: str = "swe-patch-dpo",
):
    """
    Main DPO training function.
    
    Args:
        config_path: Path to YAML config file
        config: Config object (alternative to config_path)
        dataset_path: Path to preference dataset
        wandb_project: W&B project name
    """
    # Load config
    if config is None:
        if config_path:
            config = DPOTrainingConfig.from_yaml(config_path)
        else:
            config = DPOTrainingConfig()
    
    # Initialize wandb
    wandb.init(project=wandb_project, config=vars(config))
    
    # Load model
    model, tokenizer = load_model_for_dpo(config)
    
    # Load preference dataset
    full_dataset = prepare_preference_dataset(dataset_path)
    
    # Split for evaluation
    if len(full_dataset) > 50:
        split = full_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]
    else:
        train_dataset = full_dataset
        eval_dataset = None
    
    # Create trainer
    trainer = create_dpo_trainer(
        model, tokenizer, train_dataset, eval_dataset, config
    )
    
    # Train
    print("Starting DPO training...")
    trainer.train()
    
    # Save final model
    print(f"Saving model to {config.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)
    
    # Push to hub
    if config.push_to_hub and config.hub_model_id:
        print(f"Pushing to HuggingFace Hub: {config.hub_model_id}")
        trainer.push_to_hub()
    
    wandb.finish()
    print("DPO training complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DPO Training for SWE-Bench")
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--dataset", type=str, default="./data/preference_pairs.json")
    parser.add_argument("--model", type=str, default="./outputs/sft")
    parser.add_argument("--output-dir", type=str, default="./outputs/dpo")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id", type=str)
    
    args = parser.parse_args()
    
    if args.config:
        train_dpo(config_path=args.config, dataset_path=args.dataset)
    else:
        config = DPOTrainingConfig(
            model_name=args.model,
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            beta=args.beta,
            push_to_hub=args.push_to_hub,
            hub_model_id=args.hub_model_id,
        )
        train_dpo(config=config, dataset_path=args.dataset)
