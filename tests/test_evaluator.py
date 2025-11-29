"""
Tests for evaluation module.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.evaluator import SWEBenchEvaluator, EvaluationConfig, evaluate
from evaluation.docker_runner import EvaluationResult, SWEBenchInstance
from evaluation.metrics import compute_metrics, MetricsReport


@pytest.fixture
def sample_instance():
    """Create a sample SWE-Bench instance."""
    return SWEBenchInstance(
        instance_id="test__test-1",
        repo="test/repo",
        base_commit="abc123",
        test_patch="test patch",
        patch="gold patch",
    )


@pytest.fixture
def sample_results():
    """Create sample evaluation results."""
    return [
        EvaluationResult(
            instance_id="test-1",
            passed=True,
            patch_applied=True,
            execution_time=10.5,
        ),
        EvaluationResult(
            instance_id="test-2",
            passed=False,
            patch_applied=True,
            execution_time=8.2,
            error_message="Test failed",
        ),
        EvaluationResult(
            instance_id="test-3",
            passed=False,
            patch_applied=False,
            execution_time=2.1,
            error_message="Patch did not apply",
        ),
    ]


class TestMetrics:
    def test_compute_metrics(self, sample_results):
        """Test metrics computation."""
        report = compute_metrics(sample_results)

        assert isinstance(report, MetricsReport)
        assert report.total_instances == 3
        assert report.resolved == 1
        assert report.patches_applied == 2
        assert report.resolve_rate == pytest.approx(1/3)
        assert report.avg_execution_time == pytest.approx((10.5 + 8.2 + 2.1) / 3)

    def test_compute_metrics_all_passed(self):
        """Test metrics when all tests pass."""
        results = [
            EvaluationResult("test-1", True, True, 5.0),
            EvaluationResult("test-2", True, True, 6.0),
        ]

        report = compute_metrics(results)

        assert report.total_instances == 2
        assert report.resolved == 2
        assert report.resolve_rate == 1.0

    def test_compute_metrics_all_failed(self):
        """Test metrics when all tests fail."""
        results = [
            EvaluationResult("test-1", False, True, 5.0),
            EvaluationResult("test-2", False, False, 6.0),
        ]

        report = compute_metrics(results)

        assert report.total_instances == 2
        assert report.resolved == 0
        assert report.resolve_rate == 0.0

    def test_compute_metrics_empty(self):
        """Test metrics with no results."""
        report = compute_metrics([])

        assert report.total_instances == 0
        assert report.resolved == 0
        assert report.resolve_rate == 0.0


class TestEvaluationConfig:
    def test_default_config(self):
        """Test default configuration."""
        config = EvaluationConfig()

        assert config.dataset_name == "ScaleAI/SWE-bench_Pro"
        assert config.max_workers == 4
        assert config.timeout_per_instance == 300
        assert config.use_gold_patches is False

    def test_custom_config(self):
        """Test custom configuration."""
        config = EvaluationConfig(
            dataset_name="custom-dataset",
            max_workers=8,
            max_instances=10,
            use_gold_patches=True,
        )

        assert config.dataset_name == "custom-dataset"
        assert config.max_workers == 8
        assert config.max_instances == 10
        assert config.use_gold_patches is True


class TestSWEBenchEvaluator:
    def test_load_predictions_list_format(self, tmp_path):
        """Test loading predictions in list format."""
        predictions_file = tmp_path / "predictions.json"
        predictions_data = [
            {"instance_id": "test-1", "patch": "patch 1"},
            {"instance_id": "test-2", "patch": "patch 2"},
        ]

        with open(predictions_file, "w") as f:
            json.dump(predictions_data, f)

        config = EvaluationConfig(predictions_path=str(predictions_file))
        evaluator = SWEBenchEvaluator(config)
        predictions = evaluator.load_predictions(str(predictions_file))

        assert len(predictions) == 2
        assert predictions["test-1"] == "patch 1"
        assert predictions["test-2"] == "patch 2"

    def test_load_predictions_dict_format(self, tmp_path):
        """Test loading predictions in dict format."""
        predictions_file = tmp_path / "predictions.json"
        predictions_data = {
            "test-1": "patch 1",
            "test-2": "patch 2",
        }

        with open(predictions_file, "w") as f:
            json.dump(predictions_data, f)

        config = EvaluationConfig(predictions_path=str(predictions_file))
        evaluator = SWEBenchEvaluator(config)
        predictions = evaluator.load_predictions(str(predictions_file))

        assert len(predictions) == 2
        assert predictions["test-1"] == "patch 1"
        assert predictions["test-2"] == "patch 2"

    @patch('evaluation.evaluator.load_dataset')
    def test_load_instances(self, mock_load_dataset):
        """Test loading instances."""
        mock_dataset = [
            {
                "instance_id": "test-1",
                "repo": "test/repo",
                "base_commit": "abc",
                "test_patch": "test",
                "patch": "gold",
            }
        ]
        mock_load_dataset.return_value = mock_dataset

        config = EvaluationConfig(max_instances=None)
        evaluator = SWEBenchEvaluator(config)
        instances = evaluator.load_instances()

        assert len(instances) == 1
        assert instances[0].instance_id == "test-1"

    @patch('evaluation.evaluator.load_dataset')
    def test_load_instances_with_limit(self, mock_load_dataset):
        """Test loading instances with max_instances limit."""
        mock_dataset = [
            {
                "instance_id": f"test-{i}",
                "repo": "test/repo",
                "base_commit": "abc",
                "test_patch": "test",
                "patch": "gold",
            }
            for i in range(10)
        ]
        mock_load_dataset.return_value = mock_dataset

        config = EvaluationConfig(max_instances=3)
        evaluator = SWEBenchEvaluator(config)
        instances = evaluator.load_instances()

        assert len(instances) == 3

    @patch('evaluation.evaluator.SWEBenchEvaluator.load_instances')
    @patch('evaluation.evaluator.DockerRunner')
    def test_run_evaluation(self, mock_docker_runner, mock_load_instances, sample_instance, sample_results):
        """Test running evaluation."""
        mock_load_instances.return_value = [sample_instance]

        mock_runner = MagicMock()
        mock_runner.evaluate_batch.return_value = sample_results
        mock_docker_runner.return_value = mock_runner

        predictions = {sample_instance.instance_id: "test patch"}

        config = EvaluationConfig()
        evaluator = SWEBenchEvaluator(config)
        results = evaluator.run_evaluation(
            instances=[sample_instance],
            predictions=predictions
        )

        assert len(results) == len(sample_results)


# Integration tests (require Docker)
class TestDockerIntegration:
    @pytest.mark.docker
    @pytest.mark.slow
    def test_evaluate_with_gold_patch(self):
        """Test evaluation with a gold patch (should pass)."""
        # This test requires Docker to be running
        from evaluation.docker_runner import DockerRunner
        from datasets import load_dataset

        # Load a single instance
        dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
        instance_data = dataset[0]

        instance = SWEBenchInstance(
            instance_id=instance_data["instance_id"],
            repo=instance_data["repo"],
            base_commit=instance_data["base_commit"],
            test_patch=instance_data["test_patch"],
            patch=instance_data["patch"],
        )

        runner = DockerRunner(timeout=300)
        result = runner.evaluate_instance(instance, instance.patch)

        # Gold patch should pass
        assert result.passed is True
        assert result.patch_applied is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
