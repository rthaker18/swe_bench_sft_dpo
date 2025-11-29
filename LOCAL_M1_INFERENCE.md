# Local M1 Inference Guide

This guide shows how to run inference on Apple Silicon (M1/M2/M3) Macs with 8GB RAM using smaller 3B parameter models.

## ⚠️ Important Notes

- **Training is NOT supported** on 8GB RAM - you need cloud GPUs for that
- **Inference only** works with smaller models (1-3B parameters)
- Expect **slow performance** (30-90 seconds per patch)
- This is for **testing and development**, not production use

## Requirements

- Apple Silicon Mac (M1/M2/M3)
- 8GB+ RAM (16GB recommended)
- macOS 12.0+
- Python 3.9+

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Test Inference

Test with a single instance from SWE-Bench Pro:

```bash
python scripts/run_local_inference.py
```

This will:
- Download the Qwen 3B model (~2GB, first time only)
- Load it in 4-bit quantization
- Generate a patch for one instance
- Save it to `patch_<instance_id>.diff`

Expected output:
```
================================================================================
LOADING MODEL
================================================================================
This may take 2-5 minutes on first run (downloading model)...
Expected memory usage: ~5-6GB

Loading model from: Qwen/Qwen2.5-Coder-3B-Instruct
...
================================================================================
INSTANCE 1/1
================================================================================
ID: django__django-11099
Repository: django/django
Problem: ...

Generating patch... (this may take 30-90 seconds)

--------------------------------------------------------------------------------
GENERATED PATCH:
--------------------------------------------------------------------------------
diff --git a/django/core/management/commands/...
...
```

### 3. Advanced Usage

#### Test with specific instance:

```bash
python scripts/run_local_inference.py \
    --instance-id django__django-11099
```

#### Generate predictions for multiple instances:

```bash
python scripts/run_local_inference.py \
    --max-instances 5 \
    --output predictions.json
```

#### Use a different model:

```bash
# Smaller 1.3B model (safer for 8GB RAM)
python scripts/run_local_inference.py \
    --model deepseek-ai/deepseek-coder-1.3b-instruct

# Alternative 3B model
python scripts/run_local_inference.py \
    --model bigcode/starcoder2-3b
```

#### Adjust generation parameters:

```bash
python scripts/run_local_inference.py \
    --max-new-tokens 512 \
    --temperature 0.2
```

## Recommended Models for M1 8GB

### Best Overall: Qwen 2.5 Coder 3B (Default)
```bash
--model Qwen/Qwen2.5-Coder-3B-Instruct
```
- **Size**: ~3GB in 4-bit
- **Pros**: Best code generation quality, instruction-tuned
- **Cons**: Uses ~5-6GB RAM total

### Safest Option: DeepSeek Coder 1.3B
```bash
--model deepseek-ai/deepseek-coder-1.3b-instruct
```
- **Size**: ~1.5GB in 4-bit
- **Pros**: Smallest, leaves most headroom
- **Cons**: Lower quality than 3B models

### Alternative: StarCoder2 3B
```bash
--model bigcode/starcoder2-3b
```
- **Size**: ~3GB in 4-bit
- **Pros**: Strong code generation, good for multi-language
- **Cons**: Not instruction-tuned (may need prompt tweaking)

## Performance Expectations

| Metric | Value |
|--------|-------|
| **Model download** (first time) | 2-5 minutes |
| **Model loading** | 1-3 minutes |
| **Per-patch generation** | 30-90 seconds |
| **Peak RAM usage** | 5-6GB |
| **Quality** | Good for simple patches, struggles with complex ones |

## Memory Management Tips

### If you run out of memory:

1. **Close other applications** - Free up as much RAM as possible

2. **Use a smaller model**:
   ```bash
   python scripts/run_local_inference.py \
       --model deepseek-ai/deepseek-coder-1.3b-instruct
   ```

3. **Reduce generation length**:
   ```bash
   python scripts/run_local_inference.py \
       --max-new-tokens 512
   ```

4. **Process one instance at a time**:
   ```bash
   python scripts/run_local_inference.py \
       --max-instances 1
   ```

### Monitor memory usage:

```bash
# In another terminal
watch -n 1 'ps aux | grep python'
```

Or use Activity Monitor on macOS.

## Troubleshooting

### Error: "OutOfMemoryError" or system freeze

**Solution**: Use the smaller 1.3B model:
```bash
python scripts/run_local_inference.py \
    --model deepseek-ai/deepseek-coder-1.3b-instruct
```

### Error: "Could not load model"

**Solution**: Check internet connection (model needs to download first)

### Error: "MPS backend not available"

**Solution**: The code will automatically fall back to CPU (slower but works)

### Very slow generation (>2 minutes per patch)

**Expected on M1 8GB** - This is normal. Consider:
- Using cloud GPUs for batch inference
- Using API-based inference (see below)

## Alternative: API-Based Inference

If local inference is too slow, use external APIs:

```bash
# Using OpenAI
export OPENAI_API_KEY="your-key"
python -m src.inference \
    --model gpt-4 \
    --use-api \
    --api-provider openai \
    --api-model gpt-4 \
    --max-instances 10

# Using Together AI
export TOGETHER_API_KEY="your-key"
python -m src.inference \
    --model together \
    --use-api \
    --api-provider together \
    --api-model meta-llama/Llama-3-70b-chat-hf \
    --max-instances 10
```

## What Works vs. What Doesn't

### ✅ Works on M1 8GB:
- Loading 3B models in 4-bit quantization
- Generating single patches
- Testing and development
- Data preparation and preprocessing

### ⚠️ Works but slow:
- Batch inference (process one at a time)
- Long context windows (reduce `max_new_tokens`)

### ❌ Does NOT work on M1 8GB:
- Training (SFT or DPO) - needs cloud GPUs
- 7B models - too large even with quantization
- Batch processing multiple instances simultaneously

## Next Steps

Once you've tested local inference and want to scale up:

1. **For Training**: Use cloud GPUs (RunPod/Modal) - see main [README.md](README.md)
2. **For Production Inference**: Use larger models on cloud or API services
3. **For Evaluation**: Run Docker-based evaluation (works fine on M1)

## Configuration Files

- [configs/local_inference_config.yaml](configs/local_inference_config.yaml) - M1-optimized config
- [scripts/run_local_inference.py](scripts/run_local_inference.py) - Local inference script
- [src/inference.py](src/inference.py) - Core inference code (M1-compatible)

## Resources

- [Qwen 2.5 Coder Model Card](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct)
- [DeepSeek Coder Model Card](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-instruct)
- [StarCoder2 Model Card](https://huggingface.co/bigcode/starcoder2-3b)
- [Main Project README](README.md)
