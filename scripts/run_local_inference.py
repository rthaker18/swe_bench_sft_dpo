#!/usr/bin/env python3
"""
Local inference optimized for M1 Mac with limited RAM.

This script is designed to run inference with a smaller 3B parameter model
on Apple Silicon Macs with 8GB RAM. It uses 4-bit quantization to reduce
memory usage while maintaining reasonable performance.

Usage:
    # Test with a single instance
    python scripts/run_local_inference.py

    # Test with specific instance
    python scripts/run_local_inference.py --instance-id django__django-11099

    # Generate predictions for multiple instances
    python scripts/run_local_inference.py --max-instances 10 --output predictions.json
"""

import os
import sys
import json
import argparse
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import LocalModelInference, InferenceConfig, create_prompt
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Run local inference on M1 Mac with 3B model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-Coder-3B-Instruct",
        help="Model to use (default: Qwen/Qwen2.5-Coder-3B-Instruct)",
    )
    parser.add_argument(
        "--instance-id",
        type=str,
        default=None,
        help="Specific instance ID to test (default: use first instance)",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=1,
        help="Maximum number of instances to process (default: 1)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for predictions (default: print to console)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate (default: 1024)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (default: 0.1)",
    )

    args = parser.parse_args()

    # Configuration optimized for 8GB RAM
    print(f"Configuring inference with model: {args.model}")
    config = InferenceConfig(
        model_path=args.model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=0.95,
        do_sample=True,
        load_in_4bit=True,
        device_map="auto",
    )

    print("\n" + "="*80)
    print("LOADING MODEL")
    print("="*80)
    print("This may take 2-5 minutes on first run (downloading model)...")
    print(f"Expected memory usage: ~5-6GB")
    print("")

    inference = LocalModelInference(config)

    print("\n" + "="*80)
    print("LOADING DATASET")
    print("="*80)

    dataset = load_dataset("ScaleAI/SWE-bench_Pro", split="test")
    print(f"Loaded {len(dataset)} instances from SWE-Bench Pro")

    # Select instances to process
    if args.instance_id:
        # Find specific instance
        instances = [item for item in dataset if item["instance_id"] == args.instance_id]
        if not instances:
            print(f"\nError: Instance {args.instance_id} not found in dataset")
            return
    else:
        # Use first N instances
        instances = dataset.select(range(min(args.max_instances, len(dataset))))

    print(f"\nProcessing {len(instances)} instance(s)...")

    # Generate predictions
    results = []

    for idx, sample in enumerate(instances, 1):
        print("\n" + "="*80)
        print(f"INSTANCE {idx}/{len(instances)}")
        print("="*80)
        print(f"ID: {sample['instance_id']}")
        print(f"Repository: {sample['repo']}")
        print(f"Problem: {sample['problem_statement'][:200]}...")
        print("")

        # Create prompt
        prompt = create_prompt(
            repo=sample["repo"],
            problem_statement=sample["problem_statement"],
            hints=sample.get("hints_text"),
        )

        print("Generating patch... (this may take 30-90 seconds)")

        try:
            patch = inference.generate(prompt)

            print("\n" + "-"*80)
            print("GENERATED PATCH:")
            print("-"*80)
            print(patch)
            print("-"*80)

            results.append({
                "instance_id": sample["instance_id"],
                "model_patch": patch,
            })

            # Save individual patch file
            patch_file = f"./patch_{sample['instance_id']}.diff"
            with open(patch_file, "w") as f:
                f.write(patch)
            print(f"\nPatch saved to: {patch_file}")

        except Exception as e:
            print(f"\nError generating patch: {e}")
            results.append({
                "instance_id": sample["instance_id"],
                "model_patch": "",
                "error": str(e),
            })

    # Save all results if output file specified
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n\nAll predictions saved to: {args.output}")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Processed: {len(results)} instances")
    successful = sum(1 for r in results if r.get("model_patch") and not r.get("error"))
    print(f"Successful: {successful}")
    print(f"Failed: {len(results) - successful}")
    print("="*80)


if __name__ == "__main__":
    main()
