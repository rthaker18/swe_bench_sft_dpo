"""
Data preparation utilities for SWE-Bench fine-tuning pipeline.

This module handles:
- Loading SWE-Bench Pro dataset
- Formatting data for SFT training
- Creating preference pairs for DPO
"""

import json
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from datasets import Dataset, load_dataset
import pandas as pd
from tqdm import tqdm


@dataclass
class SWEBenchInstance:
    """Represents a single SWE-Bench instance."""
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    patch: str
    test_patch: str
    hints_text: Optional[str] = None
    created_at: Optional[str] = None
    version: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict) -> "SWEBenchInstance":
        return cls(
            instance_id=data["instance_id"],
            repo=data["repo"],
            base_commit=data["base_commit"],
            problem_statement=data["problem_statement"],
            patch=data["patch"],
            test_patch=data.get("test_patch", ""),
            hints_text=data.get("hints_text"),
            created_at=data.get("created_at"),
            version=data.get("version"),
        )


@dataclass
class PreferencePair:
    """Represents a preference pair for DPO training."""
    prompt: str
    chosen: str
    rejected: str
    instance_id: str = ""


def load_swebench_pro(split: str = "test") -> List[SWEBenchInstance]:
    """
    Load SWE-Bench Pro dataset from HuggingFace.
    
    Args:
        split: Dataset split (SWE-Bench Pro only has 'test')
        
    Returns:
        List of SWEBenchInstance objects
    """
    print("Loading SWE-Bench Pro dataset...")
    dataset = load_dataset("ScaleAI/SWE-bench_Pro", split=split)
    
    instances = []
    for item in tqdm(dataset, desc="Processing instances"):
        instances.append(SWEBenchInstance.from_dict(item))
    
    print(f"Loaded {len(instances)} instances")
    return instances


def load_swebench_lite(split: str = "test") -> List[SWEBenchInstance]:
    """
    Load SWE-Bench Lite dataset (smaller, for testing).
    
    Args:
        split: Dataset split
        
    Returns:
        List of SWEBenchInstance objects
    """
    print("Loading SWE-Bench Lite dataset...")
    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split=split)
    
    instances = []
    for item in tqdm(dataset, desc="Processing instances"):
        instances.append(SWEBenchInstance.from_dict(item))
    
    print(f"Loaded {len(instances)} instances")
    return instances


def create_sft_prompt(instance: SWEBenchInstance, include_hints: bool = False) -> str:
    """
    Create a formatted prompt for SFT training.
    
    Args:
        instance: SWE-Bench instance
        include_hints: Whether to include hints in prompt
        
    Returns:
        Formatted prompt string
    """
    prompt = f"""### Task: Fix GitHub Issue

Repository: {instance.repo}

### Issue Description:
{instance.problem_statement}
"""
    
    if include_hints and instance.hints_text:
        prompt += f"""
### Hints:
{instance.hints_text}
"""
    
    prompt += """
### Instructions:
Generate a unified diff patch that fixes the issue described above.
The patch should be in standard unified diff format.

### Patch:
"""
    return prompt


def format_for_sft(
    instances: List[SWEBenchInstance],
    include_hints: bool = False,
    max_patch_length: int = 8000,
) -> Dataset:
    """
    Format instances for SFT training.
    
    Args:
        instances: List of SWE-Bench instances
        include_hints: Whether to include hints
        max_patch_length: Maximum patch length to include
        
    Returns:
        HuggingFace Dataset ready for training
    """
    formatted_data = []
    
    for instance in tqdm(instances, desc="Formatting for SFT"):
        # Skip if patch is too long
        if len(instance.patch) > max_patch_length:
            continue
            
        prompt = create_sft_prompt(instance, include_hints)
        
        formatted_data.append({
            "instance_id": instance.instance_id,
            "prompt": prompt,
            "completion": instance.patch,
            "text": prompt + instance.patch,  # Full text for causal LM training
        })
    
    print(f"Formatted {len(formatted_data)} instances for SFT")
    return Dataset.from_list(formatted_data)


def create_preference_pairs_from_evaluations(
    instances: List[SWEBenchInstance],
    evaluation_results: Dict[str, Dict],
    model_predictions: Dict[str, str],
) -> List[PreferencePair]:
    """
    Create preference pairs from evaluation results.
    
    Uses gold patches as 'chosen' and failed model predictions as 'rejected'.
    
    Args:
        instances: SWE-Bench instances
        evaluation_results: Dict mapping instance_id to eval results
        model_predictions: Dict mapping instance_id to model-generated patches
        
    Returns:
        List of PreferencePair objects
    """
    pairs = []
    
    for instance in instances:
        instance_id = instance.instance_id
        
        if instance_id not in model_predictions:
            continue
            
        result = evaluation_results.get(instance_id, {})
        model_patch = model_predictions[instance_id]
        
        # Only create pair if model prediction failed tests
        if not result.get("passed", True):
            prompt = create_sft_prompt(instance)
            pairs.append(PreferencePair(
                prompt=prompt,
                chosen=instance.patch,  # Gold patch
                rejected=model_patch,    # Failed prediction
                instance_id=instance_id,
            ))
    
    print(f"Created {len(pairs)} preference pairs")
    return pairs


def create_preference_pairs_synthetic(
    instances: List[SWEBenchInstance],
    api_client,  # OpenAI or Anthropic client
    num_negatives: int = 1,
) -> List[PreferencePair]:
    """
    Create synthetic preference pairs using an API.
    
    Generates intentionally flawed patches as negatives.
    
    Args:
        instances: SWE-Bench instances
        api_client: API client for generating negatives
        num_negatives: Number of negative examples per instance
        
    Returns:
        List of PreferencePair objects
    """
    pairs = []
    
    for instance in tqdm(instances, desc="Generating synthetic pairs"):
        prompt = create_sft_prompt(instance)
        
        # Generate a flawed patch using the API
        # This is a placeholder - implement based on your API choice
        for _ in range(num_negatives):
            try:
                flawed_patch = generate_flawed_patch(
                    api_client, 
                    instance.problem_statement,
                    instance.patch
                )
                
                pairs.append(PreferencePair(
                    prompt=prompt,
                    chosen=instance.patch,
                    rejected=flawed_patch,
                    instance_id=instance.instance_id,
                ))
            except Exception as e:
                print(f"Error generating pair for {instance.instance_id}: {e}")
                continue
    
    return pairs


def generate_flawed_patch(api_client, problem_statement: str, gold_patch: str) -> str:
    """
    Generate an intentionally flawed patch for preference training.
    
    This creates a negative example by asking the model to generate
    a plausible but incorrect patch.
    """
    # Placeholder implementation
    # In practice, you'd use Claude/GPT to generate alternatives
    prompt = f"""Given this GitHub issue:
{problem_statement}

Generate a patch that attempts to fix this issue but contains a subtle bug
or doesn't fully address the problem. The patch should look plausible but
be incorrect in some way.

Patch:"""
    
    # Call API here
    # response = api_client.complete(prompt)
    # return response
    
    raise NotImplementedError("Implement based on your API provider")


def format_for_dpo(pairs: List[PreferencePair]) -> Dataset:
    """
    Format preference pairs for DPO training.
    
    Args:
        pairs: List of PreferencePair objects
        
    Returns:
        HuggingFace Dataset in DPO format
    """
    data = []
    for pair in pairs:
        data.append({
            "prompt": pair.prompt,
            "chosen": pair.chosen,
            "rejected": pair.rejected,
        })
    
    return Dataset.from_list(data)


def save_dataset(dataset: Dataset, path: str, format: str = "json"):
    """Save dataset to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    if format == "json":
        dataset.to_json(path)
    elif format == "parquet":
        dataset.to_parquet(path)
    else:
        raise ValueError(f"Unknown format: {format}")
    
    print(f"Saved dataset to {path}")


def load_preference_pairs(path: str) -> Dataset:
    """Load preference pairs from disk."""
    if path.endswith(".json"):
        return Dataset.from_json(path)
    elif path.endswith(".parquet"):
        return Dataset.from_parquet(path)
    else:
        raise ValueError(f"Unknown file format: {path}")


# Example usage
if __name__ == "__main__":
    # Load and prepare SFT data
    instances = load_swebench_lite()  # Use Lite for testing
    sft_dataset = format_for_sft(instances)
    save_dataset(sft_dataset, "./data/sft_train.json")
    
    print(f"SFT dataset size: {len(sft_dataset)}")
    print(f"Sample: {sft_dataset[0]['text'][:500]}...")
