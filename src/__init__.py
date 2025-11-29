"""
SWE-Bench Patch Generation Fine-Tuning Pipeline

This package provides utilities for training models to generate patches
for GitHub issues using the SWE-Bench Pro dataset.

Main modules:
- data_prep: Data preparation and formatting
- sft_trainer: Supervised Fine-Tuning
- dpo_trainer: Direct Preference Optimization
- preference_gen: Preference pair generation
- inference: Model inference utilities
"""

__version__ = "0.1.0"

from .data_prep import (
    load_swebench_pro,
    load_swebench_lite,
    format_for_sft,
    format_for_dpo,
    SWEBenchInstance,
    PreferencePair,
)

from .sft_trainer import train_sft, SFTTrainingConfig
from .dpo_trainer import train_dpo, DPOTrainingConfig

__all__ = [
    "load_swebench_pro",
    "load_swebench_lite",
    "format_for_sft",
    "format_for_dpo",
    "SWEBenchInstance",
    "PreferencePair",
    "train_sft",
    "train_dpo",
    "SFTTrainingConfig",
    "DPOTrainingConfig",
]
