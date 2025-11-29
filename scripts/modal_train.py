"""
Modal deployment for serverless GPU training.

Modal.com provides serverless GPU access with pay-per-second billing.
This script allows you to run SFT and DPO training without managing infrastructure.

Usage:
    # Deploy and run SFT training
    modal run scripts/modal_train.py::sft_train
    
    # Deploy and run DPO training
    modal run scripts/modal_train.py::dpo_train

Prerequisites:
    pip install modal
    modal setup  # Authenticate
"""

import modal
import os

# Create Modal app
app = modal.App("swe-patch-trainer")

# Define the training image
training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "datasets>=2.19.0",
        "accelerate>=0.30.0",
        "peft>=0.10.0",
        "trl>=0.8.0",
        "bitsandbytes>=0.43.0",
        "sentencepiece>=0.2.0",
        "huggingface-hub>=0.23.0",
        "wandb>=0.16.0",
        "PyYAML>=6.0",
    )
    .env({
        "HF_HOME": "/cache/huggingface",
        "TRANSFORMERS_CACHE": "/cache/huggingface",
    })
)

# Persistent volume for model cache and outputs
volume = modal.Volume.from_name("swe-patch-training", create_if_missing=True)


@app.function(
    image=training_image,
    gpu="A100",  # or "A10G" for cheaper option
    timeout=86400,  # 24 hours
    volumes={"/cache": volume},
    secrets=[
        modal.Secret.from_name("huggingface-secret"),  # HF_TOKEN
        modal.Secret.from_name("wandb-secret"),  # WANDB_API_KEY
    ],
)
def sft_train(
    model_name: str = "deepseek-ai/deepseek-coder-7b-base-v1.5",
    dataset_name: str = "ScaleAI/SWE-bench_Pro",
    output_name: str = "swe-patch-sft",
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    batch_size: int = 2,
    gradient_accumulation: int = 8,
    lora_r: int = 32,
    push_to_hub: bool = True,
    hub_model_id: str = None,
):
    """
    Run SFT training on Modal's serverless GPUs.
    
    Args:
        model_name: Base model to fine-tune
        dataset_name: Dataset to train on
        output_name: Name for output model
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Per-device batch size
        gradient_accumulation: Gradient accumulation steps
        lora_r: LoRA rank
        push_to_hub: Whether to push to HuggingFace Hub
        hub_model_id: HuggingFace Hub model ID
    """
    import torch
    from datasets import load_dataset, Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig
    import wandb
    
    # Initialize wandb
    wandb.init(
        project="swe-patch-training",
        name=f"sft-{output_name}",
        config={
            "model": model_name,
            "epochs": num_epochs,
            "lr": learning_rate,
            "lora_r": lora_r,
        }
    )
    
    print(f"Loading model: {model_name}")
    
    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare for training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_r * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load and prepare dataset
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="test")
    
    # Format for training
    def format_example(example):
        prompt = f"""### Task: Fix GitHub Issue

Repository: {example['repo']}

### Issue Description:
{example['problem_statement']}

### Instructions:
Generate a unified diff patch that fixes the issue described above.

### Patch:
{example['patch']}"""
        return {"text": prompt}
    
    dataset = dataset.map(format_example)
    
    # Training config
    output_dir = f"/cache/outputs/{output_name}"
    
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="paged_adamw_32bit",
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        gradient_checkpointing=True,
        max_seq_length=4096,
        packing=False,
        report_to=["wandb"],
    )
    
    # Train
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    
    print("Starting training...")
    trainer.train()
    
    # Save
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Push to hub
    if push_to_hub:
        hub_id = hub_model_id or f"your-username/{output_name}"
        print(f"Pushing to Hub: {hub_id}")
        trainer.push_to_hub(hub_id)
    
    # Commit volume to persist
    volume.commit()
    
    wandb.finish()
    
    return {"status": "success", "output_dir": output_dir}


@app.function(
    image=training_image,
    gpu="A100",
    timeout=86400,
    volumes={"/cache": volume},
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
)
def dpo_train(
    sft_model_path: str = "/cache/outputs/swe-patch-sft",
    preference_data_path: str = "/cache/data/preferences.json",
    output_name: str = "swe-patch-dpo",
    num_epochs: int = 1,
    learning_rate: float = 5e-7,
    beta: float = 0.1,
    push_to_hub: bool = True,
    hub_model_id: str = None,
):
    """
    Run DPO training on Modal's serverless GPUs.
    
    Args:
        sft_model_path: Path to SFT model (from previous training)
        preference_data_path: Path to preference dataset
        output_name: Name for output model
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        beta: DPO beta parameter
        push_to_hub: Whether to push to HuggingFace Hub
        hub_model_id: HuggingFace Hub model ID
    """
    import torch
    import json
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training, PeftConfig
    from trl import DPOTrainer, DPOConfig
    import wandb
    
    wandb.init(
        project="swe-patch-training",
        name=f"dpo-{output_name}",
        config={
            "sft_model": sft_model_path,
            "epochs": num_epochs,
            "lr": learning_rate,
            "beta": beta,
        }
    )
    
    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    # Load SFT model
    print(f"Loading SFT model: {sft_model_path}")
    
    peft_config = PeftConfig.from_pretrained(sft_model_path)
    base_model_name = peft_config.base_model_name_or_path
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    model = PeftModel.from_pretrained(model, sft_model_path)
    model = model.merge_and_unload()
    model = prepare_model_for_kbit_training(model)
    
    # Add new LoRA for DPO
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load preference data
    print(f"Loading preference data: {preference_data_path}")
    with open(preference_data_path) as f:
        pref_data = json.load(f)
    
    dataset = Dataset.from_list(pref_data)
    
    # Training config
    output_dir = f"/cache/outputs/{output_name}"
    
    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_32bit",
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=50,
        gradient_checkpointing=True,
        max_length=4096,
        max_prompt_length=2048,
        beta=beta,
        report_to=["wandb"],
        remove_unused_columns=False,
    )
    
    # Train
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    
    print("Starting DPO training...")
    trainer.train()
    
    # Save
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Push to hub
    if push_to_hub:
        hub_id = hub_model_id or f"your-username/{output_name}"
        print(f"Pushing to Hub: {hub_id}")
        trainer.push_to_hub(hub_id)
    
    volume.commit()
    wandb.finish()
    
    return {"status": "success", "output_dir": output_dir}


@app.function(
    image=training_image,
    gpu="A10G",  # Cheaper GPU for inference
    timeout=3600,
    volumes={"/cache": volume},
)
def generate_predictions(
    model_path: str = "/cache/outputs/swe-patch-dpo",
    dataset_name: str = "ScaleAI/SWE-bench_Pro",
    output_path: str = "/cache/predictions.json",
    max_instances: int = None,
):
    """
    Generate predictions using trained model.
    """
    import torch
    import json
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel, PeftConfig
    from tqdm import tqdm
    
    print(f"Loading model: {model_path}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    
    # Load model
    peft_config = PeftConfig.from_pretrained(model_path)
    base_model_name = peft_config.base_model_name_or_path
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="test")
    
    if max_instances:
        dataset = dataset.select(range(min(max_instances, len(dataset))))
    
    # Generate
    predictions = []
    
    for item in tqdm(dataset, desc="Generating"):
        prompt = f"""### Task: Fix GitHub Issue

Repository: {item['repo']}

### Issue Description:
{item['problem_statement']}

### Instructions:
Generate a unified diff patch that fixes the issue described above.

### Patch:
"""
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3000).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.1,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        patch = tokenizer.decode(generated, skip_special_tokens=True)
        
        predictions.append({
            "instance_id": item["instance_id"],
            "patch": patch,
        })
    
    # Save
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2)
    
    volume.commit()
    
    return {"status": "success", "predictions": len(predictions), "output": output_path}


@app.local_entrypoint()
def main():
    """Main entry point for Modal CLI."""
    print("SWE-Patch Trainer on Modal")
    print("\nAvailable commands:")
    print("  modal run scripts/modal_train.py::sft_train")
    print("  modal run scripts/modal_train.py::dpo_train")
    print("  modal run scripts/modal_train.py::generate_predictions")
    print("\nExample:")
    print('  modal run scripts/modal_train.py::sft_train --model-name "deepseek-ai/deepseek-coder-7b-base-v1.5"')
