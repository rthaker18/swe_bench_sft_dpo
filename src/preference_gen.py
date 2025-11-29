"""
Preference pair generation for DPO training.

This module handles generating preference pairs from various sources:
1. Model evaluation results (real patches that passed vs failed)
2. Synthetic generation using API providers (Claude, GPT-4, etc.)
3. Rule-based negative sampling

The preference pairs are used for DPO training to align the model
toward generating patches that pass tests.
"""

import os
import json
import asyncio
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import Dataset
from tqdm import tqdm

from .data_prep import (
    SWEBenchInstance,
    PreferencePair,
    create_sft_prompt,
    format_for_dpo,
)
from .inference import load_model, generate_patch


@dataclass
class PreferenceGenConfig:
    """Configuration for preference generation."""
    method: str = "evaluation"  # "evaluation", "synthetic", "hybrid"
    num_negatives_per_instance: int = 1
    api_provider: Optional[str] = None  # "openai", "anthropic", "together"
    api_key: Optional[str] = None
    model_name: Optional[str] = None  # For generating negatives
    temperature: float = 0.7
    max_workers: int = 4


def generate_from_evaluation_results(
    instances: List[SWEBenchInstance],
    evaluation_results: Dict[str, Dict],
    model_predictions: Dict[str, str],
) -> List[PreferencePair]:
    """
    Generate preference pairs from evaluation results.

    Uses patches that passed tests as 'chosen' and patches that failed as 'rejected'.

    Args:
        instances: List of SWE-Bench instances
        evaluation_results: Dict mapping instance_id to evaluation results
        model_predictions: Dict mapping instance_id to model predictions

    Returns:
        List of PreferencePair objects
    """
    pairs = []

    for instance in tqdm(instances, desc="Creating pairs from evaluations"):
        instance_id = instance.instance_id

        if instance_id not in model_predictions:
            continue

        result = evaluation_results.get(instance_id, {})
        model_patch = model_predictions[instance_id]

        # If model prediction failed, use gold patch as chosen
        if not result.get("passed", False):
            prompt = create_sft_prompt(instance)
            pairs.append(PreferencePair(
                prompt=prompt,
                chosen=instance.patch,  # Gold patch (known to pass)
                rejected=model_patch,    # Model patch (failed)
                instance_id=instance_id,
            ))

    print(f"Generated {len(pairs)} preference pairs from evaluations")
    return pairs


def generate_with_openai(
    instances: List[SWEBenchInstance],
    config: PreferenceGenConfig,
) -> List[PreferencePair]:
    """
    Generate preference pairs using OpenAI API.

    Args:
        instances: List of SWE-Bench instances
        config: PreferenceGenConfig

    Returns:
        List of PreferencePair objects
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Install openai: pip install openai")

    client = OpenAI(api_key=config.api_key or os.environ.get("OPENAI_API_KEY"))
    pairs = []

    for instance in tqdm(instances, desc="Generating with OpenAI"):
        prompt = create_sft_prompt(instance)

        for _ in range(config.num_negatives_per_instance):
            try:
                # Generate a flawed patch
                negative_prompt = f"""{prompt}

Generate a patch that attempts to fix the issue but contains one of these flaws:
- Incomplete fix (doesn't handle edge cases)
- Incorrect logic
- Breaks existing functionality
- Doesn't follow the codebase style

The patch should look plausible but be incorrect."""

                response = client.chat.completions.create(
                    model=config.model_name or "gpt-4-turbo-preview",
                    messages=[
                        {"role": "system", "content": "You are a code generation assistant that creates patches with subtle bugs for training purposes."},
                        {"role": "user", "content": negative_prompt}
                    ],
                    temperature=config.temperature,
                    max_tokens=2048,
                )

                rejected_patch = response.choices[0].message.content.strip()

                pairs.append(PreferencePair(
                    prompt=prompt,
                    chosen=instance.patch,
                    rejected=rejected_patch,
                    instance_id=instance.instance_id,
                ))
            except Exception as e:
                print(f"Error generating for {instance.instance_id}: {e}")
                continue

    print(f"Generated {len(pairs)} preference pairs with OpenAI")
    return pairs


def generate_with_anthropic(
    instances: List[SWEBenchInstance],
    config: PreferenceGenConfig,
) -> List[PreferencePair]:
    """
    Generate preference pairs using Anthropic Claude API.

    Args:
        instances: List of SWE-Bench instances
        config: PreferenceGenConfig

    Returns:
        List of PreferencePair objects
    """
    try:
        from anthropic import Anthropic
    except ImportError:
        raise ImportError("Install anthropic: pip install anthropic")

    client = Anthropic(api_key=config.api_key or os.environ.get("ANTHROPIC_API_KEY"))
    pairs = []

    for instance in tqdm(instances, desc="Generating with Claude"):
        prompt = create_sft_prompt(instance)

        for _ in range(config.num_negatives_per_instance):
            try:
                negative_prompt = f"""{prompt}

Generate a patch that attempts to fix the issue but contains a subtle bug or incompleteness.
The patch should look plausible but be incorrect in one of these ways:
- Incomplete fix (doesn't handle all cases)
- Incorrect logic
- Side effects or breaking changes
- Style inconsistencies"""

                message = client.messages.create(
                    model=config.model_name or "claude-3-5-sonnet-20241022",
                    max_tokens=2048,
                    temperature=config.temperature,
                    messages=[
                        {"role": "user", "content": negative_prompt}
                    ]
                )

                rejected_patch = message.content[0].text.strip()

                pairs.append(PreferencePair(
                    prompt=prompt,
                    chosen=instance.patch,
                    rejected=rejected_patch,
                    instance_id=instance.instance_id,
                ))
            except Exception as e:
                print(f"Error generating for {instance.instance_id}: {e}")
                continue

    print(f"Generated {len(pairs)} preference pairs with Claude")
    return pairs


def generate_with_local_model(
    instances: List[SWEBenchInstance],
    model_path: str,
    config: PreferenceGenConfig,
) -> List[PreferencePair]:
    """
    Generate preference pairs using a local model.

    This can use the SFT checkpoint to generate imperfect patches.

    Args:
        instances: List of SWE-Bench instances
        model_path: Path to local model
        config: PreferenceGenConfig

    Returns:
        List of PreferencePair objects
    """
    print(f"Loading local model from {model_path}")
    model, tokenizer = load_model(model_path)

    pairs = []

    for instance in tqdm(instances, desc="Generating with local model"):
        prompt = create_sft_prompt(instance)

        for i in range(config.num_negatives_per_instance):
            try:
                # Generate with higher temperature for diversity
                rejected_patch = generate_patch(
                    model,
                    tokenizer,
                    prompt,
                    temperature=config.temperature + (i * 0.1),  # Increase temp for variety
                    max_new_tokens=2048,
                )

                # Only use if different from gold patch
                if rejected_patch != instance.patch:
                    pairs.append(PreferencePair(
                        prompt=prompt,
                        chosen=instance.patch,
                        rejected=rejected_patch,
                        instance_id=instance.instance_id,
                    ))
            except Exception as e:
                print(f"Error generating for {instance.instance_id}: {e}")
                continue

    print(f"Generated {len(pairs)} preference pairs with local model")
    return pairs


def generate_rule_based_negatives(
    instances: List[SWEBenchInstance],
    config: PreferenceGenConfig,
) -> List[PreferencePair]:
    """
    Generate negative examples using rule-based transformations.

    Creates plausible but incorrect patches by:
    - Removing lines from the gold patch
    - Swapping additions and deletions
    - Truncating the patch

    Args:
        instances: List of SWE-Bench instances
        config: PreferenceGenConfig

    Returns:
        List of PreferencePair objects
    """
    pairs = []

    for instance in tqdm(instances, desc="Generating rule-based negatives"):
        prompt = create_sft_prompt(instance)
        gold_patch = instance.patch

        # Strategy 1: Truncate patch (incomplete fix)
        lines = gold_patch.split('\n')
        if len(lines) > 5:
            truncated = '\n'.join(lines[:len(lines)//2])
            pairs.append(PreferencePair(
                prompt=prompt,
                chosen=gold_patch,
                rejected=truncated,
                instance_id=instance.instance_id,
            ))

        # Strategy 2: Remove some changed lines
        if len(lines) > 10:
            # Remove every 3rd line that starts with + or -
            modified_lines = []
            counter = 0
            for line in lines:
                if line.startswith(('+', '-')) and not line.startswith(('+++', '---')):
                    counter += 1
                    if counter % 3 == 0:
                        continue  # Skip this line
                modified_lines.append(line)

            if len(modified_lines) < len(lines):
                pairs.append(PreferencePair(
                    prompt=prompt,
                    chosen=gold_patch,
                    rejected='\n'.join(modified_lines),
                    instance_id=instance.instance_id,
                ))

    print(f"Generated {len(pairs)} rule-based negative pairs")
    return pairs


def generate_preference_pairs(
    instances: List[SWEBenchInstance],
    config: PreferenceGenConfig,
    evaluation_results: Optional[Dict[str, Dict]] = None,
    model_predictions: Optional[Dict[str, str]] = None,
    local_model_path: Optional[str] = None,
) -> List[PreferencePair]:
    """
    Main function to generate preference pairs.

    Args:
        instances: List of SWE-Bench instances
        config: PreferenceGenConfig
        evaluation_results: Optional evaluation results for evaluation-based generation
        model_predictions: Optional model predictions for evaluation-based generation
        local_model_path: Optional path to local model for generation

    Returns:
        List of PreferencePair objects
    """
    pairs = []

    if config.method == "evaluation":
        if evaluation_results is None or model_predictions is None:
            raise ValueError("evaluation_results and model_predictions required for evaluation method")
        pairs = generate_from_evaluation_results(instances, evaluation_results, model_predictions)

    elif config.method == "synthetic":
        if config.api_provider == "openai":
            pairs = generate_with_openai(instances, config)
        elif config.api_provider == "anthropic":
            pairs = generate_with_anthropic(instances, config)
        elif config.api_provider == "local":
            if local_model_path is None:
                raise ValueError("local_model_path required for local generation")
            pairs = generate_with_local_model(instances, local_model_path, config)
        else:
            raise ValueError(f"Unknown API provider: {config.api_provider}")

    elif config.method == "rule_based":
        pairs = generate_rule_based_negatives(instances, config)

    elif config.method == "hybrid":
        # Combine multiple methods
        if evaluation_results and model_predictions:
            pairs.extend(generate_from_evaluation_results(instances, evaluation_results, model_predictions))

        # Add rule-based negatives
        pairs.extend(generate_rule_based_negatives(instances, config))

        print(f"Generated {len(pairs)} total pairs using hybrid method")

    else:
        raise ValueError(f"Unknown method: {config.method}")

    return pairs


def save_preference_pairs(
    pairs: List[PreferencePair],
    output_path: str,
    format: str = "json",
):
    """
    Save preference pairs to disk.

    Args:
        pairs: List of PreferencePair objects
        output_path: Output file path
        format: "json" or "dataset" (HuggingFace Dataset)
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if format == "json":
        data = [
            {
                "prompt": p.prompt,
                "chosen": p.chosen,
                "rejected": p.rejected,
                "instance_id": p.instance_id,
            }
            for p in pairs
        ]
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    elif format == "dataset":
        dataset = format_for_dpo(pairs)
        dataset.save_to_disk(output_path)

    else:
        raise ValueError(f"Unknown format: {format}")

    print(f"Saved {len(pairs)} preference pairs to {output_path}")


# CLI entry point
if __name__ == "__main__":
    import argparse
    from .data_prep import load_swebench_lite, load_swebench_pro

    parser = argparse.ArgumentParser(description="Generate preference pairs for DPO")
    parser.add_argument("--method", choices=["evaluation", "synthetic", "rule_based", "hybrid"],
                       default="rule_based")
    parser.add_argument("--dataset", choices=["pro", "lite"], default="lite")
    parser.add_argument("--output", type=str, default="./data/preference_pairs.json")
    parser.add_argument("--api-provider", choices=["openai", "anthropic", "local"])
    parser.add_argument("--api-key", type=str)
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--local-model-path", type=str)
    parser.add_argument("--num-negatives", type=int, default=1)
    parser.add_argument("--max-instances", type=int, help="Limit instances for testing")
    parser.add_argument("--evaluation-results", type=str, help="Path to evaluation results JSON")
    parser.add_argument("--model-predictions", type=str, help="Path to model predictions JSON")

    args = parser.parse_args()

    # Load instances
    print(f"Loading {args.dataset} dataset...")
    if args.dataset == "pro":
        instances = load_swebench_pro()
    else:
        instances = load_swebench_lite()

    if args.max_instances:
        instances = instances[:args.max_instances]

    # Load evaluation data if provided
    eval_results = None
    model_preds = None

    if args.evaluation_results:
        with open(args.evaluation_results) as f:
            eval_data = json.load(f)
            eval_results = {r["instance_id"]: r for r in eval_data.get("results", [])}

    if args.model_predictions:
        with open(args.model_predictions) as f:
            model_preds = json.load(f)

    # Generate preference pairs
    config = PreferenceGenConfig(
        method=args.method,
        num_negatives_per_instance=args.num_negatives,
        api_provider=args.api_provider,
        api_key=args.api_key,
        model_name=args.model_name,
    )

    pairs = generate_preference_pairs(
        instances,
        config,
        evaluation_results=eval_results,
        model_predictions=model_preds,
        local_model_path=args.local_model_path,
    )

    # Save
    save_preference_pairs(pairs, args.output)
    print(f"Done! Generated {len(pairs)} preference pairs")
