"""
Quick test script to verify the trained model generates patches.
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_model(
    base_model: str = "deepseek-ai/deepseek-coder-7b-base-v1.5",
    adapter_model: str = "rahjeetee/swe-patch-sft",
    max_new_tokens: int = 1024,
):
    """Test the trained model on a sample problem."""

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading adapter from {adapter_model}...")
    model = PeftModel.from_pretrained(model, adapter_model)
    model.eval()

    # Sample test problem
    test_prompt = """### Task: Fix GitHub Issue

Repository: django/django

### Issue Description:
The `QuerySet.distinct()` method doesn't work correctly with `union()`.
When calling `qs1.union(qs2).distinct()`, the DISTINCT clause is not
applied to the final SQL query.

### Instructions:
Generate a unified diff patch that fixes the issue described above.
The patch should be in standard unified diff format.

### Patch:
"""

    print("\n" + "="*80)
    print("Testing model generation...")
    print("="*80)
    print(f"\nPrompt:\n{test_prompt}\n")

    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

    print("Generating patch...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    patch = generated[len(test_prompt):]

    print("="*80)
    print("Generated Patch:")
    print("="*80)
    print(patch)
    print("="*80)

    return patch


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="deepseek-ai/deepseek-coder-7b-base-v1.5")
    parser.add_argument("--adapter-model", default="rahjeetee/swe-patch-sft")
    parser.add_argument("--max-tokens", type=int, default=1024)

    args = parser.parse_args()

    test_model(args.base_model, args.adapter_model, args.max_tokens)
