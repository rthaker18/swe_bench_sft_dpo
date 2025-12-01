"""
Supervised Fine-Tuning (SFT) trainer for SWE-Bench patch generation.

This module provides a cost-efficient SFT implementation using:
- QLoRA for memory-efficient training
- TRL's SFTTrainer for simplified training loop
- Support for external GPU providers (RunPod, Modal)
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
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

import wandb


@dataclass
class SFTTrainingConfig:
    """Configuration for SFT training."""
    # Model
    model_name: str = "deepseek-ai/deepseek-coder-7b-base-v1.5"
    torch_dtype: str = "bfloat16"
    
    # Quantization
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    
    # LoRA
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    
    # Training
    output_dir: str = "./outputs/sft"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    max_seq_length: int = 4096
    
    # Hub
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    
    @classmethod
    def from_yaml(cls, path: str) -> "SFTTrainingConfig":
        """Load config from YAML file."""
        with open(path) as f:
            config_dict = yaml.safe_load(f)
        
        # Flatten nested config
        flat_config = {}
        for section, values in config_dict.items():
            if isinstance(values, dict):
                for k, v in values.items():
                    flat_config[k] = v
            else:
                flat_config[section] = values
        
        return cls(**{k: v for k, v in flat_config.items() 
                     if k in cls.__dataclass_fields__})


def get_bnb_config(config: SFTTrainingConfig) -> BitsAndBytesConfig:
    """Create BitsAndBytes quantization config."""
    compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)
    
    return BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=True,
    )


def get_lora_config(config: SFTTrainingConfig) -> LoraConfig:
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


def load_model_and_tokenizer(config: SFTTrainingConfig):
    """Load model with quantization and prepare for training."""
    print(f"Loading model: {config.model_name}")

    # Check GPU availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available! Training on CPU is not supported for large models.")

    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
    )

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Quantization config
    bnb_config = get_bnb_config(config)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=getattr(torch, config.torch_dtype),
    )

    print(f"Model device map: {model.hf_device_map}")

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Add LoRA adapters
    lora_config = get_lora_config(config)
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    model.print_trainable_parameters()

    return model, tokenizer


def prepare_dataset(
    dataset_path: str,
    tokenizer,
    max_seq_length: int,
    text_field: str = "text",
) -> Dataset:
    """
    Load and prepare dataset for training.
    
    Args:
        dataset_path: Path to dataset (local or HF hub)
        tokenizer: Tokenizer instance
        max_seq_length: Maximum sequence length
        text_field: Field containing the training text
        
    Returns:
        Prepared Dataset
    """
    # Load dataset
    if os.path.exists(dataset_path):
        if dataset_path.endswith(".json"):
            dataset = Dataset.from_json(dataset_path)
        else:
            dataset = Dataset.from_parquet(dataset_path)
    else:
        dataset = load_dataset(dataset_path, split="train")
    
    print(f"Loaded dataset with {len(dataset)} examples")
    return dataset


def create_trainer(
    model,
    tokenizer,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset],
    config: SFTTrainingConfig,
) -> SFTTrainer:
    """Create SFT trainer instance."""

    # Detect BF16 support
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    if use_bf16:
        print("Using BF16 mixed precision training")
    elif use_fp16:
        print("BF16 not supported, using FP16 mixed precision training")
    else:
        print("Warning: No mixed precision support detected, training will be slower")

    # Training arguments
    training_args = SFTConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        optim="paged_adamw_32bit",
        fp16=use_fp16,
        bf16=use_bf16,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=100 if eval_dataset else None,
        gradient_checkpointing=True,
        max_length=config.max_seq_length,
        packing=False,  # Don't pack sequences for code generation
        push_to_hub=config.push_to_hub,
        hub_model_id=config.hub_model_id,
        report_to=["wandb"] if wandb.run else ["tensorboard"],
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    
    return trainer


def train_sft(
    config_path: str = None,
    config: SFTTrainingConfig = None,
    dataset_path: str = "./data/sft_train.json",
    wandb_project: str = "swe-patch-sft",
):
    """
    Main training function.
    
    Args:
        config_path: Path to YAML config file
        config: Config object (alternative to config_path)
        dataset_path: Path to training dataset
        wandb_project: W&B project name
    """
    # Load config
    if config is None:
        if config_path:
            config = SFTTrainingConfig.from_yaml(config_path)
        else:
            config = SFTTrainingConfig()
    
    # Initialize wandb
    wandb.init(project=wandb_project, config=vars(config))
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Prepare dataset
    train_dataset = prepare_dataset(
        dataset_path,
        tokenizer,
        config.max_seq_length,
    )
    
    # Split for eval (optional)
    if len(train_dataset) > 100:
        split = train_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]
    else:
        eval_dataset = None
    
    # Create trainer
    trainer = create_trainer(
        model, tokenizer, train_dataset, eval_dataset, config
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print(f"Saving model to {config.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)
    
    # Push to hub if configured
    if config.push_to_hub and config.hub_model_id:
        print(f"Pushing to HuggingFace Hub: {config.hub_model_id}")
        trainer.push_to_hub()
    
    wandb.finish()
    print("Training complete!")


# For running on external providers like RunPod
def runpod_entrypoint():
    """Entry point for RunPod serverless."""
    import json
    
    # RunPod passes input via environment
    input_data = json.loads(os.environ.get("INPUT", "{}"))
    
    config = SFTTrainingConfig(
        model_name=input_data.get("model_name", "deepseek-ai/deepseek-coder-7b-base-v1.5"),
        num_train_epochs=input_data.get("epochs", 3),
        push_to_hub=input_data.get("push_to_hub", True),
        hub_model_id=input_data.get("hub_model_id"),
    )
    
    train_sft(
        config=config,
        dataset_path=input_data.get("dataset_path", "./data/sft_train.json"),
    )
    
    return {"status": "completed"}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SFT Training for SWE-Bench")
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--dataset", type=str, default="./data/sft_train.json")
    parser.add_argument("--output-dir", type=str, default="./outputs/sft")
    parser.add_argument("--model", type=str, default="deepseek-ai/deepseek-coder-7b-base-v1.5")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id", type=str)
    
    args = parser.parse_args()
    
    if args.config:
        train_sft(config_path=args.config, dataset_path=args.dataset)
    else:
        config = SFTTrainingConfig(
            model_name=args.model,
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            push_to_hub=args.push_to_hub,
            hub_model_id=args.hub_model_id,
        )
        train_sft(config=config, dataset_path=args.dataset)
