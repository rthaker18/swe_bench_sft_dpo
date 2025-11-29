"""
Inference utilities for generating patches with trained models.

Supports:
- Local model inference (with optional quantization)
- External API inference (OpenAI, Anthropic, Together)
- Batch inference for full dataset evaluation
"""

import os
import json
import time
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm


@dataclass
class InferenceConfig:
    """Configuration for inference."""
    model_path: str
    max_new_tokens: int = 2048
    temperature: float = 0.1
    top_p: float = 0.95
    do_sample: bool = True
    load_in_4bit: bool = True
    device_map: str = "auto"


class BaseInference(ABC):
    """Base class for inference backends."""
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a patch for the given prompt."""
        pass
    
    def generate_batch(
        self,
        prompts: List[str],
        instance_ids: List[str],
        show_progress: bool = True,
    ) -> Dict[str, str]:
        """
        Generate patches for multiple prompts.
        
        Args:
            prompts: List of formatted prompts
            instance_ids: Corresponding instance IDs
            show_progress: Show progress bar
            
        Returns:
            Dict mapping instance_id to generated patch
        """
        results = {}
        iterator = zip(instance_ids, prompts)
        
        if show_progress:
            iterator = tqdm(list(iterator), desc="Generating patches")
        
        for instance_id, prompt in iterator:
            try:
                patch = self.generate(prompt)
                results[instance_id] = patch
            except Exception as e:
                print(f"Error generating for {instance_id}: {e}")
                results[instance_id] = ""
        
        return results


class LocalModelInference(BaseInference):
    """
    Inference using a local model (with optional PEFT adapters).
    
    Supports QLoRA models for memory-efficient inference.
    """
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.model, self.tokenizer = self._load_model()
    
    def _load_model(self):
        """Load model with optional quantization and PEFT."""
        print(f"Loading model from: {self.config.model_path}")

        # Quantization config
        if self.config.load_in_4bit:
            # Use float16 for better M1 compatibility (fallback from bfloat16)
            compute_dtype = torch.float16
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            # Set memory limits for systems with limited RAM
            max_memory = {"cpu": "6GB"}
        else:
            bnb_config = None
            max_memory = None
        
        # Check if this is a PEFT model
        is_peft = os.path.exists(
            os.path.join(self.config.model_path, "adapter_config.json")
        )
        
        if is_peft:
            from peft import PeftConfig
            peft_config = PeftConfig.from_pretrained(self.config.model_path)
            base_model_name = peft_config.base_model_name_or_path
            
            # Load base model
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map=self.config.device_map,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                max_memory=max_memory,
                low_cpu_mem_usage=True,
            )
            
            # Load adapter
            model = PeftModel.from_pretrained(model, self.config.model_path)
            
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        else:
            # Regular model
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                quantization_config=bnb_config,
                device_map=self.config.device_map,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                max_memory=max_memory,
                low_cpu_mem_usage=True,
            )
            
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model.eval()
        return model, tokenizer
    
    def generate(self, prompt: str) -> str:
        """Generate a patch for the given prompt."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096 - self.config.max_new_tokens,
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=self.config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only the generated part
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        
        # Extract patch from response
        return self._extract_patch(response)
    
    def _extract_patch(self, response: str) -> str:
        """Extract the patch portion from model response."""
        # Look for diff markers
        if "diff --git" in response:
            # Find the start of the diff
            start = response.find("diff --git")
            return response[start:].strip()
        
        # Look for --- and +++ markers
        if "---" in response and "+++" in response:
            lines = response.split("\n")
            patch_lines = []
            in_patch = False
            
            for line in lines:
                if line.startswith("---") or line.startswith("diff --git"):
                    in_patch = True
                if in_patch:
                    patch_lines.append(line)
            
            return "\n".join(patch_lines)
        
        # Return as-is if no clear patch markers
        return response.strip()


class APIInference(BaseInference):
    """
    Inference using external APIs (OpenAI, Anthropic, Together).
    
    Useful when you don't have local GPU resources.
    """
    
    def __init__(
        self,
        provider: str,  # "openai", "anthropic", "together"
        model_name: str,
        api_key: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ):
        self.provider = provider
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Get API key from env if not provided
        if api_key is None:
            env_var = f"{provider.upper()}_API_KEY"
            api_key = os.environ.get(env_var)
            if api_key is None:
                raise ValueError(f"API key not found. Set {env_var} environment variable.")
        
        self._setup_client(api_key)
    
    def _setup_client(self, api_key: str):
        """Initialize API client."""
        if self.provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        elif self.provider == "anthropic":
            from anthropic import Anthropic
            self.client = Anthropic(api_key=api_key)
        elif self.provider == "together":
            from together import Together
            self.client = Together(api_key=api_key)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def generate(self, prompt: str) -> str:
        """Generate using API."""
        if self.provider == "openai":
            return self._generate_openai(prompt)
        elif self.provider == "anthropic":
            return self._generate_anthropic(prompt)
        elif self.provider == "together":
            return self._generate_together(prompt)
    
    def _generate_openai(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content
    
    def _generate_anthropic(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    
    def _generate_together(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content


def create_prompt(
    repo: str,
    problem_statement: str,
    hints: Optional[str] = None,
) -> str:
    """Create a formatted prompt for patch generation."""
    prompt = f"""### Task: Fix GitHub Issue

Repository: {repo}

### Issue Description:
{problem_statement}
"""
    
    if hints:
        prompt += f"""
### Hints:
{hints}
"""
    
    prompt += """
### Instructions:
Generate a unified diff patch that fixes the issue described above.
The patch should be in standard unified diff format, starting with "diff --git".

### Patch:
"""
    return prompt


def generate_predictions(
    model_path: str,
    dataset_path: str = "ScaleAI/SWE-bench_Pro",
    output_path: str = "./predictions.json",
    max_instances: Optional[int] = None,
    use_api: bool = False,
    api_provider: str = "together",
    api_model: str = "meta-llama/Llama-3-70b-chat-hf",
):
    """
    Generate predictions for the full dataset.
    
    Args:
        model_path: Path to model (or HuggingFace model ID)
        dataset_path: Path to dataset
        output_path: Where to save predictions
        max_instances: Limit number of instances
        use_api: Use API instead of local model
        api_provider: API provider if use_api
        api_model: Model name for API
    """
    from datasets import load_dataset
    
    # Load dataset
    print(f"Loading dataset: {dataset_path}")
    dataset = load_dataset(dataset_path, split="test")
    
    if max_instances:
        dataset = dataset.select(range(min(max_instances, len(dataset))))
    
    # Create inference backend
    if use_api:
        inference = APIInference(
            provider=api_provider,
            model_name=api_model,
        )
    else:
        config = InferenceConfig(model_path=model_path)
        inference = LocalModelInference(config)
    
    # Generate prompts
    prompts = []
    instance_ids = []
    
    for item in dataset:
        prompt = create_prompt(
            repo=item["repo"],
            problem_statement=item["problem_statement"],
            hints=item.get("hints_text"),
        )
        prompts.append(prompt)
        instance_ids.append(item["instance_id"])
    
    # Generate patches
    print(f"Generating patches for {len(prompts)} instances...")
    predictions = inference.generate_batch(prompts, instance_ids)
    
    # Save predictions
    output = [
        {"instance_id": k, "patch": v}
        for k, v in predictions.items()
    ]
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"Saved {len(output)} predictions to {output_path}")
    return predictions


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate SWE-Bench predictions")
    parser.add_argument("--model", type=str, required=True, help="Model path or HF ID")
    parser.add_argument("--output", type=str, default="./predictions.json")
    parser.add_argument("--dataset", type=str, default="ScaleAI/SWE-bench_Pro")
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--use-api", action="store_true")
    parser.add_argument("--api-provider", type=str, default="together")
    parser.add_argument("--api-model", type=str, default="meta-llama/Llama-3-70b-chat-hf")
    
    args = parser.parse_args()
    
    generate_predictions(
        model_path=args.model,
        dataset_path=args.dataset,
        output_path=args.output,
        max_instances=args.max_instances,
        use_api=args.use_api,
        api_provider=args.api_provider,
        api_model=args.api_model,
    )
