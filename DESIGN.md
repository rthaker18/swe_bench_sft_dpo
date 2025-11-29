# SWE-Bench Patch Generation Fine-Tuning Pipeline

## High-Level Design Overview

This document outlines a cost-efficient pipeline for training an open-source model to write patches for GitHub issues using SWE-Bench Pro dataset, without requiring local GPUs.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TRAINING PIPELINE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │   SWE-Bench  │───▶│  Data Prep   │───▶│  SFT Stage   │                  │
│  │   Pro Data   │    │   Pipeline   │    │  (RunPod/    │                  │
│  │              │    │              │    │   Modal)     │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│                                                 │                           │
│                                                 ▼                           │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │  Hugging     │◀───│  DPO Stage   │◀───│  Generate    │                  │
│  │  Face Hub    │    │  (Alignment) │    │  Preferences │                  │
│  │              │    │              │    │              │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         EVALUATION PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │  SWE-Bench   │───▶│   Docker     │───▶│   Run Tests  │                  │
│  │  Instance    │    │   Container  │    │   Compare    │                  │
│  │              │    │   (per task) │    │   Results    │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│         │                                       │                           │
│         ▼                                       ▼                           │
│  ┌──────────────┐                       ┌──────────────┐                   │
│  │  Generate    │                       │   Metrics    │                   │
│  │  Patch w/    │                       │   Report     │                   │
│  │  Model       │                       │              │                   │
│  └──────────────┘                       └──────────────┘                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Pipeline Stages

### Stage 1: Data Preparation

**Input**: SWE-Bench Pro dataset from HuggingFace
**Output**: Formatted datasets for SFT and DPO training

The SWE-Bench Pro dataset contains:
- `instance_id`: Unique identifier
- `repo`: Repository path
- `problem_statement`: The GitHub issue description
- `patch`: Gold patch that fixes the issue
- `base_commit`: Commit hash to checkout
- `test_patch`: Tests to verify the fix

**Data Transformation for SFT**:
```
prompt: "Given the following GitHub issue in repository {repo}:\n\n{problem_statement}\n\nGenerate a patch to fix this issue."
completion: {patch}
```

### Stage 2: Supervised Fine-Tuning (SFT)

**Purpose**: Teach the model the basic format and patterns of generating patches

**Cost-Efficient Approach**:
- Use QLoRA (4-bit quantization with LoRA adapters)
- Train on RunPod/Modal with pay-per-use GPUs
- Target model: CodeLlama-7B, DeepSeek-Coder-7B, or Qwen2.5-Coder-7B

**Configuration**:
- LoRA rank: 16-64
- LoRA alpha: 32-128
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- Learning rate: 2e-4
- Batch size: 4 (with gradient accumulation)
- Epochs: 3-5

### Stage 3: Preference Data Generation

**Purpose**: Generate pairs of (chosen, rejected) patches for DPO

**Approach**:
1. Use the SFT model to generate multiple patches per issue
2. Run each patch through SWE-Bench evaluation
3. Patches that pass tests → "chosen"
4. Patches that fail tests → "rejected"

**Alternative** (cheaper):
- Use external API (Claude, GPT-4) to generate alternative patches
- Gold patch = "chosen"
- Generated but incorrect patches = "rejected"

### Stage 4: Direct Preference Optimization (DPO)

**Purpose**: Align the model to prefer correct patches

**Why DPO over RLHF**:
- Simpler implementation (no reward model needed)
- More stable training
- Lower computational cost
- Works well with limited preference data

**Configuration**:
- Beta: 0.1-0.5
- Learning rate: 5e-7
- Same LoRA config as SFT

### Stage 5: Evaluation

**Environment**: Docker containers matching SWE-Bench Pro spec

**Process**:
1. Load instance (checkout repo at base_commit)
2. Generate patch using model
3. Apply patch to codebase
4. Apply test_patch
5. Run tests
6. Record pass/fail

## Cost Estimates (Approximate)

| Stage | GPU Hours | Est. Cost (RunPod A100) |
|-------|-----------|------------------------|
| SFT (7B model) | 8-16 hrs | $16-32 |
| Preference Gen | 4-8 hrs | $8-16 |
| DPO | 4-8 hrs | $8-16 |
| Evaluation | 2-4 hrs | $4-8 |
| **Total** | **18-36 hrs** | **$36-72** |

*Based on ~$2/hr for A100 40GB on RunPod*

## Directory Structure

```
swe_patch_trainer/
├── DESIGN.md                 # This document
├── requirements.txt          # Python dependencies
├── configs/
│   ├── sft_config.yaml       # SFT training configuration
│   └── dpo_config.yaml       # DPO training configuration
├── src/
│   ├── __init__.py
│   ├── data_prep.py          # Data preparation utilities
│   ├── sft_trainer.py        # SFT training script
│   ├── preference_gen.py     # Generate preference pairs
│   ├── dpo_trainer.py        # DPO training script
│   └── inference.py          # Model inference utilities
├── evaluation/
│   ├── __init__.py
│   ├── docker_runner.py      # Docker container management
│   ├── evaluator.py          # Evaluation orchestration
│   └── metrics.py            # Metrics calculation
├── scripts/
│   ├── run_sft.py            # Entry point for SFT
│   ├── run_dpo.py            # Entry point for DPO
│   ├── run_eval.py           # Entry point for evaluation
│   └── upload_to_hub.py      # Upload model to HuggingFace
└── tests/
    ├── test_data_prep.py
    ├── test_evaluator.py
    └── test_single_instance.py  # Test single SWE-Bench instance
```

## GPU Provider Options

### Option 1: RunPod (Recommended for beginners)
- Pros: Simple UI, good pricing, Docker support
- Cons: Manual container management
- Cost: ~$2/hr for A100 40GB

### Option 2: Modal
- Pros: Serverless, pay-per-second, Python-native
- Cons: Learning curve, vendor lock-in
- Cost: ~$2.78/hr for A100 40GB

### Option 3: Together AI
- Pros: API access, fine-tuning service built-in
- Cons: Less control, limited to supported models
- Cost: Varies by model

## Key Design Decisions

1. **QLoRA over Full Fine-tuning**: 10x memory reduction, minimal quality loss
2. **DPO over PPO-based RLHF**: Simpler, cheaper, more stable
3. **7B Model Size**: Good balance of capability vs. cost
4. **Docker-based Evaluation**: Matches SWE-Bench Pro's official setup
5. **Modular Design**: Each stage can run independently

## RL Environment Design

For this task, the "environment" is the SWE-Bench evaluation harness:

```python
class SWEBenchEnv:
    """
    State: (repo_state, issue_description)
    Action: patch_string
    Reward: 
        +1 if all tests pass
        0 if patch applies but tests fail
        -1 if patch doesn't apply
    Done: After single patch attempt (episodic)
    """
    
    def reset(self, instance_id):
        # Setup Docker container
        # Checkout correct commit
        return state
    
    def step(self, patch):
        # Apply patch
        # Run tests
        # Return reward
        return state, reward, done, info
```

## Next Steps

1. Set up development environment
2. Download and preprocess SWE-Bench Pro data
3. Create SFT training script
4. Set up RunPod/Modal for training
5. Train SFT model
6. Generate preference data
7. Train with DPO
8. Evaluate on full benchmark
9. Upload to HuggingFace
