#!/usr/bin/env python3
"""
Entry point for SWE-Bench evaluation.

This script provides a command-line interface for evaluating model predictions
against the SWE-Bench test suite using Docker containers.

Usage:
    python scripts/run_eval.py --predictions ./predictions.json
    python scripts/run_eval.py --predictions ./predictions.json --max-instances 10
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.evaluator import evaluate, compare_models, EvaluationConfig, SWEBenchEvaluator


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model predictions on SWE-Bench",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate predictions
  python scripts/run_eval.py --predictions ./predictions.json

  # Limit to 10 instances for testing
  python scripts/run_eval.py --predictions ./predictions.json --max-instances 10

  # Use gold patches for validation
  python scripts/run_eval.py --use-gold --max-instances 5

  # Compare base vs fine-tuned model
  python scripts/run_eval.py \\
      --predictions ./finetuned_predictions.json \\
      --compare ./base_predictions.json \\
      --output-dir ./comparison

  # Evaluate single instance
  python scripts/run_eval.py \\
      --single django__django-11099 \\
      --patch-file ./patch.diff
        """
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--single",
        type=str,
        metavar="INSTANCE_ID",
        help="Evaluate a single instance by ID"
    )
    mode_group.add_argument(
        "--use-gold",
        action="store_true",
        help="Use gold patches for validation (sanity check)"
    )

    # Data
    parser.add_argument(
        "--predictions",
        type=str,
        help="Path to predictions JSON file"
    )
    parser.add_argument(
        "--patch-file",
        type=str,
        help="Path to patch file (for --single mode)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ScaleAI/SWE-bench_Pro",
        choices=["ScaleAI/SWE-bench_Pro", "princeton-nlp/SWE-bench_Lite"],
        help="Dataset to evaluate against"
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evaluation_results",
        help="Output directory for results"
    )

    # Comparison mode
    parser.add_argument(
        "--compare",
        type=str,
        metavar="BASE_PREDICTIONS",
        help="Compare against base model predictions"
    )

    # Execution
    parser.add_argument(
        "--max-instances",
        type=int,
        help="Limit number of instances to evaluate (for testing)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of parallel evaluation workers"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout per instance in seconds"
    )

    args = parser.parse_args()

    # Single instance mode
    if args.single:
        if not args.patch_file and not args.use_gold:
            print("Error: --patch-file required for single instance evaluation")
            print("  or use --use-gold to evaluate the gold patch")
            sys.exit(1)

        print(f"Evaluating single instance: {args.single}")

        config = EvaluationConfig(
            dataset_name=args.dataset,
            output_dir=args.output_dir,
            timeout_per_instance=args.timeout,
        )

        evaluator = SWEBenchEvaluator(config)

        # Load patch
        if args.use_gold:
            instances = evaluator.load_instances()
            instance = next((i for i in instances if i.instance_id == args.single), None)
            if not instance:
                print(f"Error: Instance {args.single} not found")
                sys.exit(1)
            patch = instance.patch
            print("Using gold patch")
        else:
            with open(args.patch_file) as f:
                patch = f.read()
            print(f"Loaded patch from {args.patch_file}")

        # Evaluate
        result = evaluator.evaluate_single(args.single, patch)

        # Print result
        print("\n" + "="*70)
        print("EVALUATION RESULT")
        print("="*70)
        print(f"Instance: {result.instance_id}")
        print(f"Passed: {result.passed}")
        print(f"Patch applied: {result.patch_applied}")
        print(f"Execution time: {result.execution_time:.1f}s")
        if result.error_message:
            print(f"Error: {result.error_message}")
        print("="*70)

        sys.exit(0 if result.passed else 1)

    # Comparison mode
    if args.compare:
        if not args.predictions:
            print("Error: --predictions required for comparison mode")
            sys.exit(1)

        print(f"Comparing base ({args.compare}) vs fine-tuned ({args.predictions})")

        compare_models(
            base_predictions=args.compare,
            finetuned_predictions=args.predictions,
            output_dir=args.output_dir,
            dataset=args.dataset,
            max_workers=args.max_workers,
        )

        sys.exit(0)

    # Standard evaluation mode
    if not args.predictions and not args.use_gold:
        print("Error: --predictions required (or --use-gold for validation)")
        parser.print_help()
        sys.exit(1)

    # Print configuration
    print("\n" + "="*70)
    print("EVALUATION CONFIGURATION")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    if args.use_gold:
        print("Mode: Gold patch validation")
    else:
        print(f"Predictions: {args.predictions}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max workers: {args.max_workers}")
    print(f"Timeout per instance: {args.timeout}s")
    if args.max_instances:
        print(f"Max instances: {args.max_instances}")
    print("="*70 + "\n")

    # Confirm
    if not args.use_gold and args.max_instances is None:
        print("Warning: Full evaluation can take several hours and require significant compute.")
        response = input("Continue? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("Evaluation cancelled.")
            sys.exit(0)

    # Run evaluation
    try:
        report = evaluate(
            predictions_path=args.predictions if args.predictions else "",
            output_dir=args.output_dir,
            dataset=args.dataset,
            max_workers=args.max_workers,
            max_instances=args.max_instances,
            use_gold=args.use_gold,
        )

        print("\n" + "="*70)
        print("EVALUATION COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"Results saved to: {args.output_dir}")
        print("="*70)

    except Exception as e:
        print("\n" + "="*70)
        print("EVALUATION FAILED")
        print("="*70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
