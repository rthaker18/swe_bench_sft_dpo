#!/usr/bin/env python3
"""
Upload trained model to HuggingFace Hub.

Usage:
    python scripts/upload_to_hub.py --model-path ./outputs/dpo --repo-id your-username/swe-patch-model
"""

import os
import argparse
import json
from typing import Optional

from huggingface_hub import HfApi, create_repo, upload_folder
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig


def upload_model_to_hub(
    model_path: str,
    repo_id: str,
    private: bool = False,
    commit_message: str = "Upload fine-tuned SWE-Patch model",
):
    """
    Upload a trained model to HuggingFace Hub.
    
    Args:
        model_path: Local path to model directory
        repo_id: HuggingFace repo ID (e.g., "username/model-name")
        private: Whether to make the repo private
        commit_message: Commit message
    """
    print(f"Uploading model from {model_path} to {repo_id}")
    
    api = HfApi()
    
    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, private=private, exist_ok=True)
        print(f"Repository ready: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"Note: {e}")
    
    # Check if this is a PEFT model
    is_peft = os.path.exists(os.path.join(model_path, "adapter_config.json"))
    
    if is_peft:
        print("Detected PEFT model (LoRA adapters)")
    else:
        print("Detected full model")
    
    # Upload the model folder
    print("Uploading files...")
    api.upload_folder(
        folder_path=model_path,
        repo_id=repo_id,
        commit_message=commit_message,
    )
    
    print(f"\n✓ Upload complete!")
    print(f"  View at: https://huggingface.co/{repo_id}")
    
    # Print usage instructions
    print(f"\n{'='*60}")
    print("TO USE THIS MODEL:")
    print(f"{'='*60}")
    
    if is_peft:
        # Load the base model info
        peft_config = PeftConfig.from_pretrained(model_path)
        base_model = peft_config.base_model_name_or_path
        
        print(f"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "{base_model}",
    device_map="auto",
    torch_dtype="auto",
)

# Load your fine-tuned adapter
model = PeftModel.from_pretrained(base_model, "{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{base_model}")

# Generate
prompt = "Your prompt here"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=1024)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
""")
    else:
        print(f"""
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")

# Generate
prompt = "Your prompt here"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=1024)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
""")
    
    return repo_id


def create_model_card(
    repo_id: str,
    model_path: str,
    base_model: str,
    dataset: str = "ScaleAI/SWE-bench_Pro",
    training_method: str = "SFT + DPO",
    metrics: Optional[dict] = None,
):
    """
    Create and upload a model card.
    
    Args:
        repo_id: HuggingFace repo ID
        model_path: Local model path
        base_model: Base model used
        dataset: Training dataset
        training_method: Training method description
        metrics: Evaluation metrics
    """
    metrics_section = ""
    if metrics:
        metrics_section = f"""
## Evaluation Results

| Metric | Value |
|--------|-------|
| Resolved | {metrics.get('resolved', 'N/A')} |
| Resolve Rate | {metrics.get('resolve_rate', 'N/A')}% |
| Patches Applied | {metrics.get('patches_applied', 'N/A')} |
"""
    
    model_card = f"""---
tags:
- code
- patch-generation
- swe-bench
license: apache-2.0
datasets:
- {dataset}
base_model: {base_model}
---

# SWE-Patch: GitHub Issue Patch Generation Model

This model is fine-tuned to generate patches that fix GitHub issues, trained on the SWE-Bench Pro dataset.

## Model Description

- **Base Model**: {base_model}
- **Training Dataset**: [{dataset}](https://huggingface.co/datasets/{dataset})
- **Training Method**: {training_method}
- **Task**: Given a GitHub issue description and repository context, generate a unified diff patch to fix the issue.

## Training

This model was trained using:
1. **Supervised Fine-Tuning (SFT)**: Learning to generate patches from issue-patch pairs
2. **Direct Preference Optimization (DPO)**: Aligning the model to prefer patches that pass tests

### Training Configuration

- LoRA rank: 32 (SFT), 16 (DPO)
- Learning rate: 2e-4 (SFT), 5e-7 (DPO)
- Epochs: 3 (SFT), 1 (DPO)
- Quantization: 4-bit (QLoRA)

{metrics_section}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load model
base_model = AutoModelForCausalLM.from_pretrained("{base_model}", device_map="auto")
model = PeftModel.from_pretrained(base_model, "{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{base_model}")

# Create prompt
prompt = '''### Task: Fix GitHub Issue

Repository: django/django

### Issue Description:
[Your issue description here]

### Instructions:
Generate a unified diff patch that fixes the issue.

### Patch:
'''

# Generate
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=2048, temperature=0.1)
patch = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Limitations

- Model performance may vary across different repositories and issue types
- Generated patches should be reviewed before application
- May struggle with complex, multi-file changes

## Citation

If you use this model, please cite:

```
@misc{{swe-patch-model,
  title={{SWE-Patch: GitHub Issue Patch Generation Model}},
  year={{2024}},
  url={{https://huggingface.co/{repo_id}}}
}}
```

## Related Work

- [SWE-Bench](https://www.swebench.com/)
- [SWE-Bench Pro](https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro)
"""
    
    # Save model card
    card_path = os.path.join(model_path, "README.md")
    with open(card_path, "w") as f:
        f.write(model_card)
    
    # Upload
    api = HfApi()
    api.upload_file(
        path_or_fileobj=card_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        commit_message="Update model card",
    )
    
    print(f"Model card updated: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload model to HuggingFace Hub")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model directory")
    parser.add_argument("--repo-id", type=str, required=True, help="HuggingFace repo ID")
    parser.add_argument("--private", action="store_true", help="Make repo private")
    parser.add_argument("--base-model", type=str, default="deepseek-ai/deepseek-coder-7b-base-v1.5")
    parser.add_argument("--create-card", action="store_true", help="Create model card")
    parser.add_argument("--metrics-file", type=str, help="Path to metrics JSON file")
    
    args = parser.parse_args()
    
    # Upload model
    upload_model_to_hub(
        model_path=args.model_path,
        repo_id=args.repo_id,
        private=args.private,
    )
    
    # Create model card if requested
    if args.create_card:
        metrics = None
        if args.metrics_file and os.path.exists(args.metrics_file):
            with open(args.metrics_file) as f:
                metrics = json.load(f)
        
        create_model_card(
            repo_id=args.repo_id,
            model_path=args.model_path,
            base_model=args.base_model,
            metrics=metrics,
        )
