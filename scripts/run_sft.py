#!/usr/bin/env python3
"""
Entry point for Supervised Fine-Tuning (SFT) training.

This script provides a simple command-line interface for running SFT training
with various configuration options.

Usage:
    python scripts/run_sft.py --config configs/sft_config.yaml
    python scripts/run_sft.py --model deepseek-ai/deepseek-coder-7b-base-v1.5 --dataset ./data/sft_train.json
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sft_trainer import train_sft, SFTTrainingConfig


def main():
    parser = argparse.ArgumentParser(
        description="Train a model using Supervised Fine-Tuning on SWE-Bench data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using config file
  python scripts/run_sft.py --config configs/sft_config.yaml

  # Using command-line arguments
  python scripts/run_sft.py \\
      --model deepseek-ai/deepseek-coder-7b-base-v1.5 \\
      --dataset ./data/sft_train.json \\
      --output-dir ./outputs/sft \\
      --epochs 3 \\
      --batch-size 2

  # Push to HuggingFace Hub
  python scripts/run_sft.py \\
      --config configs/sft_config.yaml \\
      --push-to-hub \\
      --hub-model-id username/swe-patch-sft
        """
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file (overrides other arguments)"
    )

    # Data
    parser.add_argument(
        "--dataset",
        type=str,
        default="./data/sft_train.json",
        help="Path to training dataset (JSON or Parquet)"
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/deepseek-coder-7b-base-v1.5",
        help="Model name or path"
    )

    # Training
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/sft",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device training batch size"
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=8,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=4096,
        help="Maximum sequence length"
    )

    # LoRA
    parser.add_argument(
        "--lora-r",
        type=int,
        default=32,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=64,
        help="LoRA alpha"
    )

    # HuggingFace Hub
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push model to HuggingFace Hub after training"
    )
    parser.add_argument(
        "--hub-model-id",
        type=str,
        help="HuggingFace Hub model ID (e.g., username/model-name)"
    )

    # Weights & Biases
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="swe-patch-sft",
        help="W&B project name"
    )

    args = parser.parse_args()

    # Load config or create from args
    if args.config:
        print(f"Loading config from {args.config}")
        config = SFTTrainingConfig.from_yaml(args.config)

        # Override with command-line args if provided
        if args.output_dir != "./outputs/sft":
            config.output_dir = args.output_dir
        if args.push_to_hub:
            config.push_to_hub = args.push_to_hub
        if args.hub_model_id:
            config.hub_model_id = args.hub_model_id
    else:
        print("Creating config from command-line arguments")
        config = SFTTrainingConfig(
            model_name=args.model,
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            learning_rate=args.learning_rate,
            max_seq_length=args.max_seq_length,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            push_to_hub=args.push_to_hub,
            hub_model_id=args.hub_model_id,
        )

    # Verify dataset exists
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset not found at {args.dataset}")
        print("\nTo prepare the dataset, run:")
        print("  python -m src.data_prep")
        sys.exit(1)

    # Print configuration
    print("\n" + "="*70)
    print("SFT TRAINING CONFIGURATION")
    print("="*70)
    print(f"Model: {config.model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {config.output_dir}")
    print(f"Epochs: {config.num_train_epochs}")
    print(f"Batch size: {config.per_device_train_batch_size}")
    print(f"Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"Effective batch size: {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"LoRA r: {config.lora_r}")
    print(f"LoRA alpha: {config.lora_alpha}")
    print(f"Max sequence length: {config.max_seq_length}")
    print(f"Push to Hub: {config.push_to_hub}")
    if config.push_to_hub:
        print(f"Hub model ID: {config.hub_model_id}")
    print("="*70 + "\n")

    # Confirm
    response = input("Start training? [y/N]: ")
    if response.lower() not in ['y', 'yes']:
        print("Training cancelled.")
        sys.exit(0)

    # Train
    try:
        train_sft(
            config=config,
            dataset_path=args.dataset,
            wandb_project=args.wandb_project,
        )
        print("\n" + "="*70)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"Model saved to: {config.output_dir}")
        if config.push_to_hub:
            print(f"Model uploaded to: https://huggingface.co/{config.hub_model_id}")
        print("="*70)
    except Exception as e:
        print("\n" + "="*70)
        print("TRAINING FAILED")
        print("="*70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
