# SWE-Bench Evaluation Guide

This guide explains how to evaluate your trained model on SWE-Bench and publish results.

## Quick Start

### 1. Generate Predictions

Run on your GPU instance (RunPod, etc.):

```bash
python scripts/evaluate_swebench.py \
    --adapter-model rahjeetee/swe-patch-sft \
    --dataset pro \
    --output-dir ./evaluation_results
```

This will:
- Load your trained model
- Generate patches for all SWE-Bench instances
- Save predictions in the correct format
- Generate a summary report

**Options:**
- `--dataset pro` or `--dataset lite` - Which benchmark to use
- `--limit 10` - Test on just 10 instances first
- `--temperature 0.7` - Sampling temperature (lower = more deterministic)
- `--max-tokens 1024` - Max tokens to generate per patch

**Time estimate:**
- Lite (300 instances): ~30-60 minutes
- Pro (731 instances): ~2-3 hours

### 2. Run Official Evaluation

The official evaluation requires Docker and runs actual tests:

```bash
# Install SWE-Bench evaluation harness
pip install swe-bench

# Run evaluation (this takes several hours!)
python -m swebench.harness.run_evaluation \
    --predictions_path evaluation_results/predictions_pro_*.jsonl \
    --swe_bench_tasks princeton-nlp/SWE-bench_Pro \
    --log_dir ./evaluation_logs \
    --testbed ./testbed \
    --timeout 900 \
    --num_workers 4
```

**Important:**
- Requires Docker installed and running
- Each instance runs in isolated container
- Can take 8-24 hours for full evaluation
- Requires significant disk space (~50GB)

**Time estimate:**
- Lite: 4-8 hours
- Pro: 12-24 hours

### 3. Get Results

```bash
# Generate metrics report
python -m swebench.metrics.get_results \
    --log_dir ./evaluation_logs \
    --output_file ./evaluation_results/final_results.json

# Pretty print results
python -m swebench.metrics.get_results \
    --log_dir ./evaluation_logs \
    --output_file ./evaluation_results/final_results.json \
    --pretty
```

## Evaluation Metrics

SWE-Bench reports several key metrics:

### Primary Metrics

- **Resolved (%)**: Percentage of instances where the patch:
  - Applies cleanly
  - Passes all PASS_TO_PASS tests
  - Doesn't break any FAIL_TO_PASS tests

- **% Resolved**: This is the main benchmark metric

### Secondary Metrics

- **% Applied**: Patches that apply without conflicts
- **% Tests Passed**: Average test pass rate
- **% Resolved (Applied)**: Resolved rate among patches that applied

## Publishing Results

### 1. Update Model Card on Hugging Face

Add evaluation results to your model card:

```markdown
## Evaluation Results

### SWE-Bench Pro

| Metric | Score |
|--------|-------|
| Resolved | X.X% |
| Applied | X.X% |
| Tests Passed | X.X% |

Evaluated on SWE-Bench Pro (731 instances) on YYYY-MM-DD.
```

### 2. Upload to Hugging Face Leaderboard

Submit to: https://huggingface.co/spaces/princeton-nlp/SWE-bench

### 3. Create a Results Report

```bash
# The evaluation script creates a summary markdown
cat evaluation_results/predictions_pro_*_summary.md
```

### 4. Share on Social Media

Example tweet:
```
🚀 Released my SWE-Bench model!

Trained deepseek-coder-7b on SWE-Bench Pro:
✅ X.X% resolved on SWE-Bench Pro
✅ Full LoRA fine-tuning
✅ Open source

Model: https://huggingface.co/rahjeetee/swe-patch-sft
Code: https://github.com/rajatthaker/swe_bench_sft_dpo
```

## Faster Testing Workflow

Before running full evaluation:

```bash
# 1. Quick test on 10 instances
python scripts/evaluate_swebench.py \
    --adapter-model rahjeetee/swe-patch-sft \
    --dataset lite \
    --limit 10 \
    --output-dir ./evaluation_results/test

# 2. Check output quality manually
cat evaluation_results/test/predictions_*.jsonl | head -5

# 3. If looks good, run on SWE-Bench Lite (300 instances)
python scripts/evaluate_swebench.py \
    --adapter-model rahjeetee/swe-patch-sft \
    --dataset lite \
    --output-dir ./evaluation_results

# 4. Finally, run on full SWE-Bench Pro
python scripts/evaluate_swebench.py \
    --adapter-model rahjeetee/swe-patch-sft \
    --dataset pro \
    --output-dir ./evaluation_results
```

## Common Issues

### Out of Memory During Generation

Reduce batch size or use CPU offloading:

```bash
# Use 8-bit quantization
python scripts/evaluate_swebench.py \
    --adapter-model rahjeetee/swe-patch-sft \
    --load-in-8bit
```

### Docker Issues

Make sure Docker is running:
```bash
sudo systemctl start docker  # Linux
# or start Docker Desktop on Mac/Windows
```

### Evaluation Takes Too Long

Run in parallel with more workers:
```bash
python -m swebench.harness.run_evaluation \
    --num_workers 8 \
    ...
```

Or run on a subset first:
```bash
--swe_bench_tasks princeton-nlp/SWE-bench_Lite
```

## Cost Estimates

**Inference (prediction generation):**
- RunPod RTX 4090: ~$2-5 for full Pro dataset
- RunPod A100: ~$5-10 for full Pro dataset

**Evaluation (running tests):**
- Can run on any machine with Docker (no GPU needed)
- CPU-bound, takes 12-24 hours
- Can run on cheaper CPU instances

**Total cost:** ~$5-15 for complete evaluation

## Citation

If you publish results, cite SWE-Bench:

```bibtex
@article{jimenez2024swebench,
  title={SWE-bench: Can Language Models Resolve Real-world Github Issues?},
  author={Jimenez, Carlos E and Yang, John and Wettig, Alexander and Yao, Shunyu and Pei, Kexin and Press, Ofir and Narasimhan, Karthik},
  journal={arXiv preprint arXiv:2310.06770},
  year={2024}
}
```
