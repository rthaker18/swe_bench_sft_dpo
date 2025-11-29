# Data Directory

This directory stores training and evaluation data for the SWE-Bench patch generation pipeline.

## Expected Files

After running the data preparation scripts, you should have:

- `sft_train.json` - Formatted training data for SFT
- `preference_pairs.json` - Preference pairs for DPO training
- `predictions.json` - Model predictions for evaluation

## Generating Data

### SFT Training Data

```bash
python -m src.data_prep
```

This will:
1. Load SWE-Bench Pro dataset
2. Format it for SFT training
3. Save to `data/sft_train.json`

### Preference Pairs

```bash
# Using rule-based generation (cheapest)
python -m src.preference_gen \
    --method rule_based \
    --dataset lite \
    --output ./data/preference_pairs.json

# Using API-based generation (higher quality)
python -m src.preference_gen \
    --method synthetic \
    --api-provider anthropic \
    --dataset lite \
    --output ./data/preference_pairs.json
```

### Model Predictions

```bash
python -m src.inference \
    --model ./outputs/dpo \
    --dataset lite \
    --output ./data/predictions.json
```

## File Formats

### SFT Training Data

```json
[
  {
    "instance_id": "django__django-11099",
    "prompt": "### Task: Fix GitHub Issue\n\n...",
    "completion": "--- a/file.py\n+++ b/file.py\n...",
    "text": "### Task: Fix GitHub Issue\n\n...--- a/file.py\n..."
  }
]
```

### Preference Pairs

```json
[
  {
    "prompt": "### Task: Fix GitHub Issue\n\n...",
    "chosen": "--- a/file.py\n+++ b/file.py\n... (correct patch)",
    "rejected": "--- a/file.py\n+++ b/file.py\n... (incorrect patch)"
  }
]
```

### Predictions

```json
{
  "django__django-11099": "--- a/file.py\n+++ b/file.py\n...",
  "django__django-11100": "--- a/file.py\n+++ b/file.py\n..."
}
```

## Notes

- Files in this directory are gitignored by default (too large)
- Use SWE-Bench Lite for development and testing
- Use SWE-Bench Pro for final evaluation
