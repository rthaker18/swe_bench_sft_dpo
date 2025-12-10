"""
Run inference on SWE-Bench dataset and generate predictions.
"""

import os
import sys
import json
import torch
from typing import List, Dict
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_prep import load_swebench_pro, create_sft_prompt


def generate_predictions(
    instances: List,
    model,
    tokenizer,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    batch_size: int = 1,
) -> Dict[str, str]:
    """
    Generate patch predictions for SWE-Bench instances.

    Args:
        instances: List of SWEBenchInstance objects
        model: Loaded model
        tokenizer: Tokenizer
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature
        batch_size: Batch size (keep at 1 for large models)

    Returns:
        Dict mapping instance_id to generated patch
    """
    predictions = {}

    model.eval()

    for instance in tqdm(instances, desc="Generating predictions"):
        prompt = create_sft_prompt(instance, include_hints=False)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3072)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        patch = generated[len(prompt):].strip()

        predictions[instance.instance_id] = patch

    return predictions


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run inference on SWE-Bench")
    parser.add_argument("--base-model", default="deepseek-ai/deepseek-coder-7b-base-v1.5")
    parser.add_argument("--adapter-model", required=True, help="HF hub model ID or local path")
    parser.add_argument("--output-file", default="./predictions/predictions.json")
    parser.add_argument("--dataset", choices=["pro", "lite"], default="pro")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--limit", type=int, help="Limit number of instances for testing")

    args = parser.parse_args()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading adapter from {args.adapter_model}...")
    model = PeftModel.from_pretrained(model, args.adapter_model)

    print(f"Loading SWE-Bench {args.dataset.upper()} dataset...")
    if args.dataset == "pro":
        from src.data_prep import load_swebench_pro
        instances = load_swebench_pro()
    else:
        from src.data_prep import load_swebench_lite
        instances = load_swebench_lite()

    if args.limit:
        instances = instances[:args.limit]
        print(f"Limited to {args.limit} instances")

    print(f"Generating predictions for {len(instances)} instances...")
    predictions = generate_predictions(
        instances,
        model,
        tokenizer,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    # Save predictions
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(predictions, f, indent=2)

    print(f"\nPredictions saved to {args.output_file}")
    print(f"Generated {len(predictions)} predictions")

    # Show sample
    if predictions:
        sample_id = list(predictions.keys())[0]
        print(f"\nSample prediction for {sample_id}:")
        print("="*80)
        print(predictions[sample_id][:500])
        if len(predictions[sample_id]) > 500:
            print("...(truncated)")
        print("="*80)


if __name__ == "__main__":
    main()
