"""
Full SWE-Bench evaluation pipeline.

This script:
1. Generates predictions for all instances
2. Formats them for SWE-Bench evaluation
3. Runs the evaluation harness
4. Generates a results report
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import List, Dict
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_prep import load_swebench_pro, load_swebench_lite, create_sft_prompt


def generate_predictions(
    instances: List,
    model,
    tokenizer,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.95,
    repetition_penalty: float = 1.0,
) -> Dict[str, str]:
    """Generate patch predictions for all instances."""
    predictions = {}

    model.eval()

    for instance in tqdm(instances, desc="Generating predictions"):
        prompt = create_sft_prompt(instance, include_hints=False)

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=3072
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        patch = generated[len(prompt):].strip()

        predictions[instance.instance_id] = patch

    return predictions


def format_for_swebench(
    instances: List,
    predictions: Dict[str, str],
    model_name: str,
) -> List[Dict]:
    """
    Format predictions in SWE-Bench evaluation format.

    Expected format:
    {
        "instance_id": "...",
        "model_patch": "...",
        "model_name_or_path": "..."
    }
    """
    formatted = []

    for instance in instances:
        instance_id = instance.instance_id
        if instance_id in predictions:
            formatted.append({
                "instance_id": instance_id,
                "model_patch": predictions[instance_id],
                "model_name_or_path": model_name,
            })

    return formatted


def save_predictions(predictions: List[Dict], output_file: str):
    """Save predictions in JSONL format for SWE-Bench."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')

    print(f"Saved {len(predictions)} predictions to {output_file}")


def generate_results_summary(
    predictions: List[Dict],
    output_file: str,
    model_name: str,
    dataset: str,
):
    """Generate a summary markdown report."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    summary = f"""# SWE-Bench Evaluation Results

**Model**: {model_name}
**Dataset**: SWE-Bench {dataset.upper()}
**Generated**: {timestamp}
**Total Instances**: {len(predictions)}

## Model Configuration

- Base Model: deepseek-ai/deepseek-coder-7b-base-v1.5
- Training: Supervised Fine-Tuning (SFT) with LoRA
- Dataset: SWE-Bench Pro training split
- Training Epochs: 3

## Predictions Generated

Total predictions generated: {len(predictions)}

## Sample Predictions

"""

    # Add samples
    for i, pred in enumerate(predictions[:3]):
        patch = pred['model_patch']
        truncated = patch[:300] + "..." if len(patch) > 300 else patch
        summary += f"""
### Instance {i+1}: {pred['instance_id']}

```diff
{truncated}
```

---
"""

    summary += f"""
## Next Steps

To evaluate these predictions:

```bash
# Install SWE-Bench evaluation harness
pip install swe-bench

# Run evaluation (requires Docker)
python -m swebench.harness.run_evaluation \\
    --predictions_path {output_file} \\
    --swe_bench_tasks {dataset} \\
    --log_dir ./evaluation_logs \\
    --testbed ./testbed \\
    --skip_existing \\
    --timeout 900 \\
    --num_workers 4

# Generate final report
python -m swebench.metrics.get_results \\
    --log_dir ./evaluation_logs
```

## Evaluation Metrics

After running the evaluation harness, you'll get:

- **Resolved (%)**: Percentage of instances where the patch passes all tests
- **Applied (%)**: Percentage of patches that apply cleanly
- **Test Pass Rate**: Individual test pass rates per instance

## Publication

Results will be published to:
- Hugging Face Model Card: https://huggingface.co/{model_name}
- Paper/Blog post (optional)
"""

    summary_file = output_file.replace('.jsonl', '_summary.md')
    with open(summary_file, 'w') as f:
        f.write(summary)

    print(f"\nResults summary saved to {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Full SWE-Bench evaluation pipeline"
    )
    parser.add_argument(
        "--base-model",
        default="deepseek-ai/deepseek-coder-7b-base-v1.5",
        help="Base model to use"
    )
    parser.add_argument(
        "--adapter-model",
        required=True,
        help="Adapter model (HF hub ID or local path)"
    )
    parser.add_argument(
        "--dataset",
        choices=["pro", "lite"],
        default="pro",
        help="Which SWE-Bench dataset to evaluate on"
    )
    parser.add_argument(
        "--output-dir",
        default="./evaluation_results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Max tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Repetition penalty (1.0 = no penalty, >1.0 = penalize repetition)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of instances (for testing)"
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip generation and use existing predictions"
    )

    args = parser.parse_args()

    # Setup output paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(
        args.output_dir,
        f"predictions_{args.dataset}_{timestamp}.jsonl"
    )

    if not args.skip_generation:
        print("="*80)
        print("Loading Model")
        print("="*80)

        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model,
            trust_remote_code=True
        )

        print("Loading base model...")
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        print(f"Loading adapter from {args.adapter_model}...")
        model = PeftModel.from_pretrained(model, args.adapter_model)
        model.eval()

        print("\n" + "="*80)
        print("Loading Dataset")
        print("="*80)

        if args.dataset == "pro":
            instances = load_swebench_pro()
        else:
            instances = load_swebench_lite()

        if args.limit:
            instances = instances[:args.limit]
            print(f"Limited to {args.limit} instances for testing")

        print(f"\nTotal instances: {len(instances)}")

        print("\n" + "="*80)
        print("Generating Predictions")
        print("="*80)

        predictions_dict = generate_predictions(
            instances,
            model,
            tokenizer,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
        )

        print("\n" + "="*80)
        print("Formatting for SWE-Bench")
        print("="*80)

        formatted_predictions = format_for_swebench(
            instances,
            predictions_dict,
            args.adapter_model,
        )

        save_predictions(formatted_predictions, output_file)
    else:
        print("Skipping generation, using existing predictions...")
        # Load existing predictions
        with open(output_file, 'r') as f:
            formatted_predictions = [json.loads(line) for line in f]

    print("\n" + "="*80)
    print("Generating Summary Report")
    print("="*80)

    generate_results_summary(
        formatted_predictions,
        output_file,
        args.adapter_model,
        args.dataset,
    )

    print("\n" + "="*80)
    print("Evaluation Complete!")
    print("="*80)
    print(f"\nPredictions saved to: {output_file}")
    print(f"Summary report: {output_file.replace('.jsonl', '_summary.md')}")
    print("\nTo evaluate these predictions, follow the instructions in the summary report.")
    print("This requires Docker and will run tests in isolated environments.")


if __name__ == "__main__":
    main()
