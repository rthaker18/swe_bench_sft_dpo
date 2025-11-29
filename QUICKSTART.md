# Quick Start Guide

Get started with SWE-Patch Trainer in minutes.

## Prerequisites

- Python 3.8+
- Docker (for evaluation)
- GPU access (local or cloud provider)

## Installation

```bash
# Clone the repository
cd swe_patch_trainer

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

## Step 1: Prepare Data (5 minutes)

Start with SWE-Bench Lite for faster experimentation:

```bash
# Generate SFT training data
python -c "
from src.data_prep import load_swebench_lite, format_for_sft, save_dataset
instances = load_swebench_lite()
dataset = format_for_sft(instances[:100])  # Use subset for testing
save_dataset(dataset, './data/sft_train.json')
"
```

## Step 2: Train SFT Model

### Option A: Local GPU

```bash
python scripts/run_sft.py \
    --model deepseek-ai/deepseek-coder-7b-base-v1.5 \
    --dataset ./data/sft_train.json \
    --output-dir ./outputs/sft \
    --epochs 3 \
    --batch-size 2
```

### Option B: Modal (Serverless)

```bash
# Setup Modal
pip install modal
modal setup

# Deploy and train
modal run scripts/modal_train.py::sft_train
```

### Option C: RunPod

```bash
# Follow RunPod instructions
python scripts/runpod_train.py --instructions
```

## Step 3: Generate Preference Pairs

Choose a method based on your budget:

### Free: Rule-Based

```bash
python -m src.preference_gen \
    --method rule_based \
    --dataset lite \
    --output ./data/preference_pairs.json
```

### Paid: API-Based (Higher Quality)

```bash
# Using Claude
export ANTHROPIC_API_KEY=your_key
python -m src.preference_gen \
    --method synthetic \
    --api-provider anthropic \
    --dataset lite \
    --output ./data/preference_pairs.json
```

## Step 4: Train DPO Model

```bash
python scripts/run_dpo.py \
    --model ./outputs/sft \
    --dataset ./data/preference_pairs.json \
    --output-dir ./outputs/dpo \
    --epochs 1 \
    --beta 0.1
```

## Step 5: Generate Predictions

```bash
python -m src.inference \
    --model ./outputs/dpo \
    --dataset lite \
    --output ./data/predictions.json \
    --max-instances 10
```

## Step 6: Evaluate

### Quick Test (Single Instance)

```bash
python scripts/run_eval.py \
    --single django__django-11099 \
    --use-gold
```

### Full Evaluation

```bash
python scripts/run_eval.py \
    --predictions ./data/predictions.json \
    --max-instances 10
```

## Expected Timeline

| Step | Time (Local GPU) | Time (Cloud) | Cost |
|------|-----------------|--------------|------|
| Data Prep | 5 min | 5 min | Free |
| SFT Training | 8-16 hrs | 8-16 hrs | $16-32 |
| Preference Gen | 1-4 hrs | 1-4 hrs | $0-20 |
| DPO Training | 4-8 hrs | 4-8 hrs | $8-16 |
| Evaluation | 2-4 hrs | 2-4 hrs | $4-8 |
| **Total** | **15-32 hrs** | **15-32 hrs** | **$28-76** |

## Tips for Success

1. **Start Small**: Use `--max-instances 10` for testing
2. **Use Lite Dataset**: SWE-Bench Lite is faster for development
3. **Monitor Training**: Use W&B for real-time monitoring
4. **Save Checkpoints**: Training can be interrupted and resumed
5. **Test Evaluation First**: Run `--use-gold --max-instances 5` to verify setup

## Common Issues

### Out of Memory

```bash
# Reduce batch size
--batch-size 1 --gradient-accumulation 32
```

### Docker Issues

```bash
# Verify Docker is running
docker info

# Test with single instance
python scripts/run_eval.py --single django__django-11099 --use-gold
```

### Model Not Found

```bash
# Verify model path
ls ./outputs/sft

# Or use HuggingFace model directly
--model deepseek-ai/deepseek-coder-7b-base-v1.5
```

## Next Steps

- Read [DESIGN.md](DESIGN.md) for architecture details
- Check [README.md](README.md) for comprehensive documentation
- Explore configuration files in `configs/`
- Run tests: `pytest tests/ -v`

## Getting Help

- Issues: [GitHub Issues](https://github.com/YOUR_USERNAME/swe_patch_trainer/issues)
- Documentation: See README.md and DESIGN.md
- Examples: Check `scripts/` and `tests/` directories
