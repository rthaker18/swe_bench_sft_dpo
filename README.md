# SWE-Bench SFT/DPO

A cost-efficient pipeline for fine-tuning open-source models to write patches for GitHub issues, using the SWE-Bench Pro dataset.

## Overview

This project implements a complete training and evaluation pipeline:

1. **SFT (Supervised Fine-Tuning)**: Train the model on issue-patch pairs
2. **DPO (Direct Preference Optimization)**: Align the model to prefer patches that pass tests
3. **Evaluation**: Docker-based evaluation matching SWE-Bench Pro's official setup

### Why This Approach?

- **Cost-efficient**: Uses QLoRA (4-bit quantization + LoRA) to train 7B models on a single GPU
- **No local GPU required**: Designed for cloud GPU providers like RunPod and Modal
- **DPO over RLHF**: Simpler, cheaper, and more stable than PPO-based RLHF
- **Reproducible evaluation**: Docker containers match SWE-Bench Pro's official environment

> **💻 Running on M1 Mac with 8GB RAM?** Check out [LOCAL_M1_INFERENCE.md](LOCAL_M1_INFERENCE.md) for instructions on running inference with smaller 3B models. Note: Training requires cloud GPUs.

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/rthaker18/swe_bench_sft_dpo.git
cd swe_bench_sft_dpo
pip install -r requirements.txt
```

### 2. Prepare Data

```python
from src.data_prep import load_swebench_pro, format_for_sft, save_dataset

# Load and format data
instances = load_swebench_pro()
dataset = format_for_sft(instances)
save_dataset(dataset, "./data/sft_train.json")
```

### 3. Train (Choose Your GPU Provider)

#### Option A: Modal (Serverless)

```bash
# Install and setup Modal
pip install modal
modal setup

# Run training
modal run scripts/modal_train.py::sft_train
```

#### Option B: RunPod (Persistent)

```bash
# Get instructions
python scripts/runpod_train.py --instructions

# Then follow the manual deployment steps
```

#### Option C: Local GPU

```bash
python -m src.sft_trainer \
    --model deepseek-ai/deepseek-coder-7b-base-v1.5 \
    --dataset ./data/sft_train.json \
    --output-dir ./outputs/sft \
    --epochs 3
```

### 4. Evaluate

```bash
# Validate setup with gold patches
python -m tests.test_single_instance --validate-gold 5

# Evaluate your model
python -m evaluation.evaluator \
    --predictions ./predictions.json \
    --output-dir ./results
```

### 5. Upload to HuggingFace

```bash
python scripts/upload_to_hub.py \
    --model-path ./outputs/dpo \
    --repo-id your-username/swe-patch-model \
    --create-card
```

## Project Structure

```
swe_bench_sft_dpo/
├── DESIGN.md                 # Detailed design documentation
├── requirements.txt          # Python dependencies
├── configs/
│   ├── sft_config.yaml       # SFT training configuration
│   └── dpo_config.yaml       # DPO training configuration
├── src/
│   ├── data_prep.py          # Data preparation utilities
│   ├── sft_trainer.py        # SFT training script
│   ├── dpo_trainer.py        # DPO training script
│   └── inference.py          # Model inference utilities
├── evaluation/
│   ├── docker_runner.py      # Docker container management
│   ├── evaluator.py          # Evaluation orchestration
│   └── metrics.py            # Metrics calculation
├── scripts/
│   ├── modal_train.py        # Modal serverless deployment
│   ├── runpod_train.py       # RunPod deployment
│   └── upload_to_hub.py      # Upload to HuggingFace
└── tests/
    └── test_single_instance.py  # Single instance testing
```

## Training Pipeline

### Stage 1: Supervised Fine-Tuning (SFT)

Teaches the model to generate patches from issue descriptions:

```
Input: Issue description + repository info
Output: Unified diff patch
```

Configuration:
- **Model**: DeepSeek-Coder-7B (or CodeLlama-7B, Qwen2.5-Coder-7B)
- **Method**: QLoRA (4-bit quantization + LoRA rank 32)
- **Duration**: ~8-16 hours on A100

### Stage 2: Direct Preference Optimization (DPO)

Aligns the model to prefer patches that pass tests:

```
Input: Prompt + chosen patch (passes tests) + rejected patch (fails tests)
Output: Model that prefers generating correct patches
```

Why DPO over RLHF:
- No reward model needed (cheaper)
- Single training loop (simpler)
- More stable training
- Works well with limited preference data

### Stage 3: Evaluation

Uses Docker containers matching SWE-Bench Pro's official setup:

1. Pull pre-built Docker image for each instance
2. Apply model-generated patch
3. Apply test patch
4. Run tests
5. Record pass/fail

## Cost Estimates

| Stage | GPU Hours | Cost (RunPod A100) |
|-------|-----------|-------------------|
| SFT | 8-16 hrs | $16-32 |
| Preference Gen | 4-8 hrs | $8-16 |
| DPO | 4-8 hrs | $8-16 |
| Evaluation | 2-4 hrs | $4-8 |
| **Total** | **18-36 hrs** | **$36-72** |

*Based on ~$2/hr for A100 40GB on RunPod*

## GPU Provider Comparison

| Provider | Pricing | Best For |
|----------|---------|----------|
| **RunPod** | ~$2/hr A100 | Beginners, full control |
| **Modal** | ~$2.78/hr A100 | Serverless, Python-native |
| **Together AI** | Pay-per-token | API-based fine-tuning |

## Configuration Options

### Model Choices

| Model | Size | Strengths |
|-------|------|-----------|
| DeepSeek-Coder-7B | 7B | Best for code, recent |
| CodeLlama-7B | 7B | Meta's code model |
| Qwen2.5-Coder-7B | 7B | Good multilingual |
| StarCoder2-7B | 7B | BigCode Foundation |

### Training Parameters

```yaml
# configs/sft_config.yaml
lora:
  r: 32           # LoRA rank (16-64)
  lora_alpha: 64  # Usually 2x rank
  lora_dropout: 0.05

training:
  learning_rate: 2e-4
  num_train_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  max_seq_length: 4096
```

## Evaluation

### Validate Setup

```bash
# Test that gold patches pass (sanity check)
python -m tests.test_single_instance --validate-gold 5
```

### Test Single Instance

```bash
# Test with your model
python -m tests.test_single_instance \
    -i "django__django-11099" \
    --model ./outputs/sft

# Test with gold patch
python -m tests.test_single_instance \
    -i "django__django-11099" \
    --use-gold
```

### Full Evaluation

```bash
# Generate predictions
python -m src.inference \
    --model ./outputs/dpo \
    --output predictions.json

# Evaluate
python -m evaluation.evaluator \
    --predictions predictions.json \
    --output-dir results

# Compare base vs fine-tuned
python -m evaluation.evaluator \
    --predictions finetuned_predictions.json \
    --compare base_predictions.json
```

## Troubleshooting

### Docker Issues

```bash
# Verify Docker is running
docker info

# Test SWE-Bench image
docker pull jefzda/sweap-images:sample_image_tag
```

### Out of Memory

- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps`
- Use a smaller LoRA rank

### Training Not Converging

- Check learning rate (try 1e-4 to 5e-4 for SFT)
- Ensure data is properly formatted
- Monitor loss on W&B

## Contributing

Contributions welcome! Please read DESIGN.md for architecture details.

## License

MIT License

## Citation

```bibtex
@software{swe_bench_sft_dpo,
  title={SWE-Bench SFT/DPO: Fine-tuning Pipeline for GitHub Issue Resolution},
  year={2025},
  url={https://github.com/rthaker18/swe_bench_sft_dpo}
}
```

## Acknowledgments

- [SWE-Bench](https://www.swebench.com/) for the benchmark and evaluation harness
- [SWE-Bench Pro](https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro) for the dataset
- [TRL](https://github.com/huggingface/trl) for the training library
- [PEFT](https://github.com/huggingface/peft) for parameter-efficient fine-tuning
