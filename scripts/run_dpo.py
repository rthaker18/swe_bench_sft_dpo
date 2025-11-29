#!/usr/bin/env python3
"""
Entry point for Direct Preference Optimization (DPO) training.

This script provides a simple command-line interface for running DPO training
to align models with preference data.

Usage:
    python scripts/run_dpo.py --config configs/dpo_config.yaml
    python scripts/run_dpo.py --model ./outputs/sft --dataset ./data/preference_pairs.json
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dpo_trainer import train_dpo, DPOTrainingConfig


def main():
    parser = argparse.ArgumentParser(
        description="Train a model using Direct Preference Optimization (DPO)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using config file
  python scripts/run_dpo.py --config configs/dpo_config.yaml

  # Using command-line arguments
  python scripts/run_dpo.py \\
      --model ./outputs/sft \\
      --dataset ./data/preference_pairs.json \\
      --output-dir ./outputs/dpo \\
      --epochs 1 \\
      --beta 0.1

  # Push to HuggingFace Hub
  python scripts/run_dpo.py \\
      --config configs/dpo_config.yaml \\
      --push-to-hub \\
      --hub-model-id username/swe-patch-dpo
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
        default="./data/preference_pairs.json",
        help="Path to preference pairs dataset (JSON or Parquet)"
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="./outputs/sft",
        help="Path to SFT model (or HuggingFace model name)"
    )

    # Training
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/dpo",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Per-device training batch size"
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=16,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-7,
        help="Learning rate"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=4096,
        help="Maximum sequence length"
    )

    # DPO-specific
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="DPO beta parameter (controls strength of alignment)"
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
        default="swe-patch-dpo",
        help="W&B project name"
    )

    args = parser.parse_args()

    # Load config or create from args
    if args.config:
        print(f"Loading config from {args.config}")
        config = DPOTrainingConfig.from_yaml(args.config)

        # Override with command-line args if provided
        if args.output_dir != "./outputs/dpo":
            config.output_dir = args.output_dir
        if args.push_to_hub:
            config.push_to_hub = args.push_to_hub
        if args.hub_model_id:
            config.hub_model_id = args.hub_model_id
    else:
        print("Creating config from command-line arguments")
        config = DPOTrainingConfig(
            model_name=args.model,
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            learning_rate=args.learning_rate,
            max_seq_length=args.max_seq_length,
            beta=args.beta,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            push_to_hub=args.push_to_hub,
            hub_model_id=args.hub_model_id,
        )

    # Verify dataset exists
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset not found at {args.dataset}")
        print("\nTo generate preference pairs, run:")
        print("  python -m src.preference_gen --method rule_based --output ./data/preference_pairs.json")
        sys.exit(1)

    # Verify model exists
    if not os.path.exists(args.model) and not args.model.startswith(("hf://", "https://")):
        print(f"Warning: Model path {args.model} not found locally.")
        response = input("Continue anyway? (will try to download from HuggingFace) [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            sys.exit(1)

    # Print configuration
    print("\n" + "="*70)
    print("DPO TRAINING CONFIGURATION")
    print("="*70)
    print(f"Base model: {config.model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {config.output_dir}")
    print(f"Epochs: {config.num_train_epochs}")
    print(f"Batch size: {config.per_device_train_batch_size}")
    print(f"Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"Effective batch size: {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Beta: {config.beta}")
    print(f"LoRA r: {config.lora_r}")
    print(f"LoRA alpha: {config.lora_alpha}")
    print(f"Max sequence length: {config.max_seq_length}")
    print(f"Push to Hub: {config.push_to_hub}")
    if config.push_to_hub:
        print(f"Hub model ID: {config.hub_model_id}")
    print("="*70 + "\n")

    # Confirm
    response = input("Start DPO training? [y/N]: ")
    if response.lower() not in ['y', 'yes']:
        print("Training cancelled.")
        sys.exit(0)

    # Train
    try:
        train_dpo(
            config=config,
            dataset_path=args.dataset,
            wandb_project=args.wandb_project,
        )
        print("\n" + "="*70)
        print("DPO TRAINING COMPLETED SUCCESSFULLY!")
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
