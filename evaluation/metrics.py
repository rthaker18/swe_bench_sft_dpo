"""
Metrics calculation for SWE-Bench evaluation.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from .docker_runner import EvaluationResult


@dataclass
class MetricsReport:
    """Summary metrics for an evaluation run."""
    total_instances: int = 0
    resolved: int = 0
    resolved_ids: List[str] = field(default_factory=list)
    patches_applied: int = 0
    patches_failed: int = 0
    
    # Test statistics
    total_tests_passed: int = 0
    total_tests_failed: int = 0
    total_tests_error: int = 0
    
    # Derived metrics
    resolve_rate: float = 0.0
    patch_apply_rate: float = 0.0
    
    # Timing
    avg_execution_time: float = 0.0
    total_execution_time: float = 0.0
    
    # Per-repo breakdown
    per_repo_resolved: Dict[str, int] = field(default_factory=dict)
    per_repo_total: Dict[str, int] = field(default_factory=dict)
    
    # Error analysis
    error_types: Dict[str, int] = field(default_factory=dict)


def compute_metrics(results: List[EvaluationResult]) -> MetricsReport:
    """
    Compute metrics from evaluation results.
    
    Args:
        results: List of EvaluationResult from evaluation run
        
    Returns:
        MetricsReport with computed metrics
    """
    report = MetricsReport()
    
    if not results:
        return report
    
    report.total_instances = len(results)
    
    per_repo_resolved = defaultdict(int)
    per_repo_total = defaultdict(int)
    error_types = defaultdict(int)
    
    for result in results:
        # Extract repo from instance_id (format: repo_owner__repo_name-PR_number)
        parts = result.instance_id.split("-")
        repo = parts[0].replace("__", "/") if parts else "unknown"
        per_repo_total[repo] += 1
        
        if result.passed:
            report.resolved += 1
            report.resolved_ids.append(result.instance_id)
            per_repo_resolved[repo] += 1
        
        if result.patch_applied:
            report.patches_applied += 1
        else:
            report.patches_failed += 1
        
        report.total_tests_passed += result.tests_passed
        report.total_tests_failed += result.tests_failed
        report.total_tests_error += result.tests_error
        
        report.total_execution_time += result.execution_time
        
        # Categorize errors
        if result.error_message:
            if "patch" in result.error_message.lower():
                error_types["patch_apply_failed"] += 1
            elif "timeout" in result.error_message.lower():
                error_types["timeout"] += 1
            elif "docker" in result.error_message.lower():
                error_types["docker_error"] += 1
            else:
                error_types["other"] += 1
    
    # Compute rates
    report.resolve_rate = report.resolved / report.total_instances if report.total_instances > 0 else 0
    report.patch_apply_rate = report.patches_applied / report.total_instances if report.total_instances > 0 else 0
    report.avg_execution_time = report.total_execution_time / report.total_instances if report.total_instances > 0 else 0
    
    report.per_repo_resolved = dict(per_repo_resolved)
    report.per_repo_total = dict(per_repo_total)
    report.error_types = dict(error_types)
    
    return report


def print_detailed_report(report: MetricsReport):
    """Print a detailed text report."""
    print("\n" + "="*60)
    print("DETAILED EVALUATION REPORT")
    print("="*60)
    
    print(f"\n{'OVERALL METRICS':-^60}")
    print(f"  Total Instances:     {report.total_instances}")
    print(f"  Resolved:            {report.resolved} ({report.resolve_rate:.1%})")
    print(f"  Patches Applied:     {report.patches_applied} ({report.patch_apply_rate:.1%})")
    print(f"  Patches Failed:      {report.patches_failed}")
    
    print(f"\n{'TEST STATISTICS':-^60}")
    print(f"  Total Tests Passed:  {report.total_tests_passed}")
    print(f"  Total Tests Failed:  {report.total_tests_failed}")
    print(f"  Total Tests Error:   {report.total_tests_error}")
    
    print(f"\n{'TIMING':-^60}")
    print(f"  Total Time:          {report.total_execution_time:.1f}s")
    print(f"  Average per Instance: {report.avg_execution_time:.1f}s")
    
    if report.per_repo_resolved:
        print(f"\n{'PER-REPO BREAKDOWN':-^60}")
        print(f"  {'Repository':<30} {'Resolved':<10} {'Total':<10} {'Rate':<10}")
        print(f"  {'-'*55}")
        for repo in sorted(report.per_repo_total.keys()):
            resolved = report.per_repo_resolved.get(repo, 0)
            total = report.per_repo_total[repo]
            rate = resolved / total if total > 0 else 0
            print(f"  {repo:<30} {resolved:<10} {total:<10} {rate:.1%}")
    
    if report.error_types:
        print(f"\n{'ERROR BREAKDOWN':-^60}")
        for error_type, count in sorted(report.error_types.items()):
            print(f"  {error_type:<30} {count}")
    
    if report.resolved_ids:
        print(f"\n{'RESOLVED INSTANCES':-^60}")
        for instance_id in report.resolved_ids[:20]:  # Show first 20
            print(f"  ✓ {instance_id}")
        if len(report.resolved_ids) > 20:
            print(f"  ... and {len(report.resolved_ids) - 20} more")
    
    print("\n" + "="*60)


def compare_reports(
    base_report: MetricsReport,
    finetuned_report: MetricsReport,
) -> Dict:
    """
    Compare two evaluation reports.
    
    Args:
        base_report: Base model evaluation report
        finetuned_report: Fine-tuned model evaluation report
        
    Returns:
        Dict with comparison metrics
    """
    comparison = {
        "resolved_improvement": finetuned_report.resolved - base_report.resolved,
        "resolve_rate_improvement": finetuned_report.resolve_rate - base_report.resolve_rate,
        "patch_apply_improvement": finetuned_report.patches_applied - base_report.patches_applied,
        
        # New instances solved by fine-tuned but not base
        "new_resolved": [
            id for id in finetuned_report.resolved_ids
            if id not in base_report.resolved_ids
        ],
        
        # Instances solved by base but not fine-tuned (regressions)
        "regressions": [
            id for id in base_report.resolved_ids
            if id not in finetuned_report.resolved_ids
        ],
        
        # Both solved
        "both_resolved": [
            id for id in base_report.resolved_ids
            if id in finetuned_report.resolved_ids
        ],
    }
    
    comparison["net_improvement"] = (
        len(comparison["new_resolved"]) - len(comparison["regressions"])
    )
    
    return comparison


def generate_leaderboard_entry(
    report: MetricsReport,
    model_name: str,
    method_name: str = "Direct Generation",
) -> Dict:
    """
    Generate a leaderboard-style entry.
    
    Args:
        report: Evaluation report
        model_name: Name of the model
        method_name: Method/approach name
        
    Returns:
        Dict formatted for leaderboard
    """
    return {
        "model": model_name,
        "method": method_name,
        "resolved": report.resolved,
        "total": report.total_instances,
        "resolve_rate": round(report.resolve_rate * 100, 2),
        "patch_apply_rate": round(report.patch_apply_rate * 100, 2),
    }
