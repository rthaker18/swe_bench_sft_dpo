#!/usr/bin/env python3
"""
Test script for evaluating a single SWE-Bench instance.

This script is useful for:
- Verifying your setup is working
- Testing individual patches before full evaluation
- Debugging evaluation issues

Usage:
    python -m tests.test_single_instance --instance-id "django__django-11099"
    python -m tests.test_single_instance --use-gold  # Test with gold patch
"""

import os
import sys
import json
import argparse
from typing import Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from evaluation.docker_runner import DockerRunner, SWEBenchInstance, EvaluationResult
from evaluation.metrics import print_detailed_report, compute_metrics
from src.inference import LocalModelInference, APIInference, create_prompt, InferenceConfig


def get_instance(
    instance_id: str,
    dataset_name: str = "ScaleAI/SWE-bench_Pro",
) -> Optional[SWEBenchInstance]:
    """Load a specific instance from the dataset."""
    dataset = load_dataset(dataset_name, split="test")
    
    for item in dataset:
        if item["instance_id"] == instance_id:
            return SWEBenchInstance(
                instance_id=item["instance_id"],
                repo=item["repo"],
                base_commit=item["base_commit"],
                test_patch=item.get("test_patch", ""),
                patch=item["patch"],
            )
    
    return None


def list_instances(
    dataset_name: str = "ScaleAI/SWE-bench_Pro",
    limit: int = 20,
) -> None:
    """List available instances."""
    dataset = load_dataset(dataset_name, split="test")
    
    print(f"\nAvailable instances in {dataset_name}:")
    print("-" * 60)
    
    for i, item in enumerate(dataset):
        if i >= limit:
            print(f"... and {len(dataset) - limit} more")
            break
        print(f"  {item['instance_id']}")


def test_single_instance(
    instance_id: str,
    patch: Optional[str] = None,
    use_gold: bool = False,
    model_path: Optional[str] = None,
    use_api: bool = False,
    api_provider: str = "together",
    api_model: str = "meta-llama/Llama-3-70b-chat-hf",
    timeout: int = 300,
    dataset_name: str = "ScaleAI/SWE-bench_Pro",
) -> EvaluationResult:
    """
    Test evaluation on a single instance.
    
    Args:
        instance_id: Instance to test
        patch: Patch to apply (optional if use_gold or model_path provided)
        use_gold: Use the gold patch
        model_path: Generate patch using this model
        use_api: Use API for generation
        api_provider: API provider
        api_model: API model name
        timeout: Evaluation timeout
        dataset_name: Dataset to use
        
    Returns:
        EvaluationResult
    """
    # Get instance
    instance = get_instance(instance_id, dataset_name)
    if instance is None:
        raise ValueError(f"Instance not found: {instance_id}")
    
    print(f"\n{'='*60}")
    print(f"Testing instance: {instance_id}")
    print(f"Repository: {instance.repo}")
    print(f"Commit: {instance.base_commit}")
    print(f"{'='*60}")
    
    # Determine patch to use
    if patch:
        test_patch = patch
        print("\nUsing provided patch")
    elif use_gold:
        test_patch = instance.patch
        print("\nUsing gold patch")
    elif model_path or use_api:
        print("\nGenerating patch...")
        test_patch = generate_patch(
            instance,
            model_path=model_path,
            use_api=use_api,
            api_provider=api_provider,
            api_model=api_model,
        )
    else:
        raise ValueError("Must provide patch, --use-gold, --model, or --use-api")
    
    # Show patch preview
    print(f"\nPatch preview (first 500 chars):")
    print("-" * 40)
    print(test_patch[:500])
    if len(test_patch) > 500:
        print(f"... ({len(test_patch) - 500} more chars)")
    print("-" * 40)
    
    # Run evaluation
    print("\nRunning evaluation...")
    runner = DockerRunner(timeout=timeout)
    result = runner.evaluate_instance(instance, test_patch)
    
    # Print result
    print(f"\n{'='*60}")
    print("RESULT")
    print(f"{'='*60}")
    print(f"  Passed:         {'✓' if result.passed else '✗'}")
    print(f"  Patch Applied:  {'✓' if result.patch_applied else '✗'}")
    print(f"  Tests Passed:   {result.tests_passed}")
    print(f"  Tests Failed:   {result.tests_failed}")
    print(f"  Tests Error:    {result.tests_error}")
    print(f"  Execution Time: {result.execution_time:.1f}s")
    
    if result.error_message:
        print(f"\n  Error: {result.error_message}")
    
    if result.logs:
        print(f"\n  Logs (last 1000 chars):")
        print("-" * 40)
        print(result.logs[-1000:])
    
    print(f"{'='*60}\n")
    
    return result


def generate_patch(
    instance: SWEBenchInstance,
    model_path: Optional[str] = None,
    use_api: bool = False,
    api_provider: str = "together",
    api_model: str = "meta-llama/Llama-3-70b-chat-hf",
) -> str:
    """Generate a patch for an instance."""
    # Create prompt
    # Note: We don't have problem_statement in our minimal instance
    # In real usage, you'd load the full dataset item
    from datasets import load_dataset
    dataset = load_dataset("ScaleAI/SWE-bench_Pro", split="test")
    
    full_item = None
    for item in dataset:
        if item["instance_id"] == instance.instance_id:
            full_item = item
            break
    
    if full_item is None:
        raise ValueError(f"Could not find instance: {instance.instance_id}")
    
    prompt = create_prompt(
        repo=full_item["repo"],
        problem_statement=full_item["problem_statement"],
        hints=full_item.get("hints_text"),
    )
    
    # Create inference backend
    if use_api:
        inference = APIInference(
            provider=api_provider,
            model_name=api_model,
        )
    elif model_path:
        config = InferenceConfig(model_path=model_path)
        inference = LocalModelInference(config)
    else:
        raise ValueError("Must provide model_path or use_api")
    
    return inference.generate(prompt)


def test_gold_patches(
    dataset_name: str = "ScaleAI/SWE-bench_Pro",
    num_instances: int = 5,
    timeout: int = 300,
):
    """
    Test that gold patches pass (validates evaluation setup).
    
    This is a sanity check - gold patches should always pass.
    """
    dataset = load_dataset(dataset_name, split="test")
    
    runner = DockerRunner(timeout=timeout)
    results = []
    
    print(f"\nValidating gold patches on {num_instances} instances...")
    print("="*60)
    
    for i, item in enumerate(dataset):
        if i >= num_instances:
            break
        
        instance = SWEBenchInstance(
            instance_id=item["instance_id"],
            repo=item["repo"],
            base_commit=item["base_commit"],
            test_patch=item.get("test_patch", ""),
            patch=item["patch"],
        )
        
        print(f"\n[{i+1}/{num_instances}] {instance.instance_id}")
        
        result = runner.evaluate_instance(instance, instance.patch)
        results.append(result)
        
        status = "✓ PASS" if result.passed else "✗ FAIL"
        print(f"  Result: {status}")
        
        if not result.passed:
            print(f"  Error: {result.error_message}")
    
    # Summary
    report = compute_metrics(results)
    
    print("\n" + "="*60)
    print("GOLD PATCH VALIDATION SUMMARY")
    print("="*60)
    print(f"  Passed: {report.resolved}/{report.total_instances}")
    print(f"  Rate:   {report.resolve_rate:.1%}")
    
    if report.resolve_rate < 1.0:
        print("\n⚠️  Warning: Some gold patches failed!")
        print("  This may indicate Docker image or evaluation issues.")
    else:
        print("\n✓ All gold patches passed - evaluation setup is working!")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test single SWE-Bench instance")
    
    parser.add_argument(
        "--instance-id", "-i",
        type=str,
        help="Instance ID to test",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available instances",
    )
    parser.add_argument(
        "--use-gold",
        action="store_true",
        help="Use gold patch (for validation)",
    )
    parser.add_argument(
        "--patch-file",
        type=str,
        help="Path to file containing patch",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model path for generating patch",
    )
    parser.add_argument(
        "--use-api",
        action="store_true",
        help="Use API for generating patch",
    )
    parser.add_argument(
        "--api-provider",
        type=str,
        default="together",
        choices=["openai", "anthropic", "together"],
    )
    parser.add_argument(
        "--api-model",
        type=str,
        default="meta-llama/Llama-3-70b-chat-hf",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ScaleAI/SWE-bench_Pro",
        help="Dataset to use",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout per instance in seconds",
    )
    parser.add_argument(
        "--validate-gold",
        type=int,
        metavar="N",
        help="Validate N gold patches (sanity check)",
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_instances(args.dataset)
    elif args.validate_gold:
        test_gold_patches(
            dataset_name=args.dataset,
            num_instances=args.validate_gold,
            timeout=args.timeout,
        )
    elif args.instance_id:
        patch = None
        if args.patch_file:
            with open(args.patch_file) as f:
                patch = f.read()
        
        test_single_instance(
            instance_id=args.instance_id,
            patch=patch,
            use_gold=args.use_gold,
            model_path=args.model,
            use_api=args.use_api,
            api_provider=args.api_provider,
            api_model=args.api_model,
            timeout=args.timeout,
            dataset_name=args.dataset,
        )
    else:
        parser.print_help()
        print("\n\nExamples:")
        print("  # List available instances")
        print("  python -m tests.test_single_instance --list")
        print("")
        print("  # Test with gold patch (validation)")
        print("  python -m tests.test_single_instance -i django__django-11099 --use-gold")
        print("")
        print("  # Validate gold patches work (sanity check)")
        print("  python -m tests.test_single_instance --validate-gold 5")
        print("")
        print("  # Test with your model")
        print("  python -m tests.test_single_instance -i django__django-11099 --model ./outputs/sft")
