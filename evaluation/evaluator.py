"""
Main evaluation orchestrator for SWE-Bench.

Coordinates:
- Loading instances and predictions
- Running Docker-based evaluations
- Computing metrics
- Generating reports
"""

import os
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from datasets import load_dataset
from tqdm import tqdm

from .docker_runner import DockerRunner, EvaluationResult, SWEBenchInstance
from .metrics import compute_metrics, MetricsReport


@dataclass
class EvaluationConfig:
    """Configuration for evaluation run."""
    dataset_name: str = "ScaleAI/SWE-bench_Pro"
    predictions_path: str = "./predictions.json"
    output_dir: str = "./evaluation_results"
    max_workers: int = 4
    timeout_per_instance: int = 300
    max_instances: Optional[int] = None  # Limit for testing
    use_gold_patches: bool = False  # For validation


class SWEBenchEvaluator:
    """
    Main evaluator class for SWE-Bench.
    
    Handles the full evaluation pipeline:
    1. Load dataset and predictions
    2. Run evaluations in Docker containers
    3. Compute metrics
    4. Generate report
    """
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.docker_runner = DockerRunner(
            timeout=config.timeout_per_instance,
        )
        
        os.makedirs(config.output_dir, exist_ok=True)
    
    def load_instances(self) -> List[SWEBenchInstance]:
        """Load SWE-Bench instances from HuggingFace."""
        print(f"Loading dataset: {self.config.dataset_name}")
        
        dataset = load_dataset(self.config.dataset_name, split="test")
        
        instances = []
        for item in dataset:
            instances.append(SWEBenchInstance(
                instance_id=item["instance_id"],
                repo=item["repo"],
                base_commit=item["base_commit"],
                test_patch=item.get("test_patch", ""),
                patch=item["patch"],  # Gold patch
            ))
        
        if self.config.max_instances:
            instances = instances[:self.config.max_instances]
        
        print(f"Loaded {len(instances)} instances")
        return instances
    
    def load_predictions(self, path: str) -> Dict[str, str]:
        """
        Load model predictions from JSON file.
        
        Expected format:
        [
            {"instance_id": "...", "patch": "..."},
            ...
        ]
        or:
        {"instance_id": "patch", ...}
        """
        with open(path) as f:
            data = json.load(f)
        
        if isinstance(data, list):
            predictions = {item["instance_id"]: item["patch"] for item in data}
        else:
            predictions = data
        
        print(f"Loaded {len(predictions)} predictions")
        return predictions
    
    def run_evaluation(
        self,
        instances: Optional[List[SWEBenchInstance]] = None,
        predictions: Optional[Dict[str, str]] = None,
    ) -> List[EvaluationResult]:
        """
        Run full evaluation.
        
        Args:
            instances: Optional pre-loaded instances
            predictions: Optional pre-loaded predictions
            
        Returns:
            List of EvaluationResult
        """
        # Load data if not provided
        if instances is None:
            instances = self.load_instances()
        
        if predictions is None:
            if self.config.use_gold_patches:
                # Use gold patches for validation
                predictions = {inst.instance_id: inst.patch for inst in instances}
            else:
                predictions = self.load_predictions(self.config.predictions_path)
        
        # Filter instances with predictions
        instances_to_eval = [
            inst for inst in instances
            if inst.instance_id in predictions
        ]
        
        print(f"Evaluating {len(instances_to_eval)} instances")
        
        # Run evaluation
        results = self.docker_runner.evaluate_batch(
            instances_to_eval,
            predictions,
            max_workers=self.config.max_workers,
        )
        
        return results
    
    def generate_report(
        self,
        results: List[EvaluationResult],
        save: bool = True,
    ) -> MetricsReport:
        """
        Generate evaluation report.
        
        Args:
            results: List of evaluation results
            save: Whether to save report to disk
            
        Returns:
            MetricsReport
        """
        # Compute metrics
        report = compute_metrics(results)
        
        if save:
            # Save detailed results
            results_file = os.path.join(
                self.config.output_dir,
                f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            with open(results_file, "w") as f:
                json.dump({
                    "config": asdict(self.config),
                    "metrics": asdict(report),
                    "results": [asdict(r) for r in results],
                }, f, indent=2)
            
            print(f"Saved results to {results_file}")
            
            # Print summary
            print("\n" + "="*50)
            print("EVALUATION SUMMARY")
            print("="*50)
            print(f"Total instances: {report.total_instances}")
            print(f"Resolved: {report.resolved} ({report.resolve_rate:.1%})")
            print(f"Patches applied: {report.patches_applied}")
            print(f"Average execution time: {report.avg_execution_time:.1f}s")
            print("="*50 + "\n")
        
        return report
    
    def evaluate_single(
        self,
        instance_id: str,
        patch: str,
    ) -> EvaluationResult:
        """
        Evaluate a single instance (useful for testing).
        
        Args:
            instance_id: Instance ID to evaluate
            patch: Patch to apply
            
        Returns:
            EvaluationResult
        """
        instances = self.load_instances()
        
        instance = next(
            (i for i in instances if i.instance_id == instance_id),
            None
        )
        
        if instance is None:
            raise ValueError(f"Instance not found: {instance_id}")
        
        return self.docker_runner.evaluate_instance(instance, patch)


def evaluate(
    predictions_path: str,
    output_dir: str = "./evaluation_results",
    dataset: str = "ScaleAI/SWE-bench_Pro",
    max_workers: int = 4,
    max_instances: Optional[int] = None,
    use_gold: bool = False,
):
    """
    Convenience function to run evaluation.
    
    Args:
        predictions_path: Path to predictions JSON
        output_dir: Output directory for results
        dataset: Dataset name
        max_workers: Number of parallel workers
        max_instances: Maximum instances to evaluate
        use_gold: Use gold patches (for validation)
    """
    config = EvaluationConfig(
        dataset_name=dataset,
        predictions_path=predictions_path,
        output_dir=output_dir,
        max_workers=max_workers,
        max_instances=max_instances,
        use_gold_patches=use_gold,
    )
    
    evaluator = SWEBenchEvaluator(config)
    results = evaluator.run_evaluation()
    report = evaluator.generate_report(results)
    
    return report


def compare_models(
    base_predictions: str,
    finetuned_predictions: str,
    output_dir: str = "./comparison_results",
    dataset: str = "ScaleAI/SWE-bench_Pro",
    max_workers: int = 4,
):
    """
    Compare base model vs fine-tuned model.
    
    Args:
        base_predictions: Path to base model predictions
        finetuned_predictions: Path to fine-tuned model predictions
        output_dir: Output directory
        dataset: Dataset name
        max_workers: Number of parallel workers
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load instances once
    config = EvaluationConfig(dataset_name=dataset)
    evaluator = SWEBenchEvaluator(config)
    instances = evaluator.load_instances()
    
    # Evaluate base model
    print("\n" + "="*50)
    print("EVALUATING BASE MODEL")
    print("="*50)
    
    base_preds = evaluator.load_predictions(base_predictions)
    base_results = evaluator.docker_runner.evaluate_batch(
        [i for i in instances if i.instance_id in base_preds],
        base_preds,
        max_workers=max_workers,
    )
    base_report = compute_metrics(base_results)
    
    # Evaluate fine-tuned model
    print("\n" + "="*50)
    print("EVALUATING FINE-TUNED MODEL")
    print("="*50)
    
    ft_preds = evaluator.load_predictions(finetuned_predictions)
    ft_results = evaluator.docker_runner.evaluate_batch(
        [i for i in instances if i.instance_id in ft_preds],
        ft_preds,
        max_workers=max_workers,
    )
    ft_report = compute_metrics(ft_results)
    
    # Comparison
    print("\n" + "="*50)
    print("COMPARISON")
    print("="*50)
    print(f"{'Metric':<25} {'Base':<15} {'Fine-tuned':<15} {'Improvement':<15}")
    print("-"*70)
    print(f"{'Resolved':<25} {base_report.resolved:<15} {ft_report.resolved:<15} {ft_report.resolved - base_report.resolved:+d}")
    print(f"{'Resolve Rate':<25} {base_report.resolve_rate:.1%:<15} {ft_report.resolve_rate:.1%:<15} {(ft_report.resolve_rate - base_report.resolve_rate)*100:+.1f}%")
    print(f"{'Patches Applied':<25} {base_report.patches_applied:<15} {ft_report.patches_applied:<15}")
    print("="*50)
    
    # Save comparison
    comparison = {
        "base_model": {
            "predictions_path": base_predictions,
            "metrics": asdict(base_report),
        },
        "finetuned_model": {
            "predictions_path": finetuned_predictions,
            "metrics": asdict(ft_report),
        },
        "improvement": {
            "resolved_delta": ft_report.resolved - base_report.resolved,
            "resolve_rate_delta": ft_report.resolve_rate - base_report.resolve_rate,
        }
    }
    
    with open(os.path.join(output_dir, "comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)
    
    return base_report, ft_report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SWE-Bench Evaluation")
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./evaluation_results")
    parser.add_argument("--dataset", type=str, default="ScaleAI/SWE-bench_Pro")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--use-gold", action="store_true", help="Use gold patches for validation")
    parser.add_argument("--compare", type=str, help="Path to base model predictions for comparison")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_models(
            args.compare,
            args.predictions,
            args.output_dir,
            args.dataset,
            args.max_workers,
        )
    else:
        evaluate(
            args.predictions,
            args.output_dir,
            args.dataset,
            args.max_workers,
            args.max_instances,
            args.use_gold,
        )
