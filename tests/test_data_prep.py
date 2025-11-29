"""
Tests for data preparation module.
"""

import pytest
from datasets import Dataset

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_prep import (
    SWEBenchInstance,
    PreferencePair,
    create_sft_prompt,
    format_for_sft,
    format_for_dpo,
)


@pytest.fixture
def sample_instance():
    """Create a sample SWE-Bench instance for testing."""
    return SWEBenchInstance(
        instance_id="test__test-1",
        repo="test/repo",
        base_commit="abc123",
        problem_statement="Fix the bug in the code",
        patch="--- a/file.py\n+++ b/file.py\n@@ -1,1 +1,1 @@\n-old\n+new",
        test_patch="--- a/test.py\n+++ b/test.py\n@@ -1,1 +1,1 @@\n-old_test\n+new_test",
        hints_text="Look at the function definition",
    )


class TestSWEBenchInstance:
    def test_from_dict(self):
        """Test creating instance from dict."""
        data = {
            "instance_id": "test-1",
            "repo": "test/repo",
            "base_commit": "abc",
            "problem_statement": "Fix bug",
            "patch": "diff",
            "test_patch": "test diff",
        }
        instance = SWEBenchInstance.from_dict(data)

        assert instance.instance_id == "test-1"
        assert instance.repo == "test/repo"
        assert instance.patch == "diff"


class TestPromptFormatting:
    def test_create_sft_prompt_basic(self, sample_instance):
        """Test basic prompt creation."""
        prompt = create_sft_prompt(sample_instance, include_hints=False)

        assert "test/repo" in prompt
        assert "Fix the bug in the code" in prompt
        assert "Patch:" in prompt
        assert "Hints:" not in prompt

    def test_create_sft_prompt_with_hints(self, sample_instance):
        """Test prompt creation with hints."""
        prompt = create_sft_prompt(sample_instance, include_hints=True)

        assert "test/repo" in prompt
        assert "Fix the bug in the code" in prompt
        assert "Hints:" in prompt
        assert "Look at the function definition" in prompt


class TestDataFormatting:
    def test_format_for_sft(self, sample_instance):
        """Test SFT dataset formatting."""
        dataset = format_for_sft([sample_instance])

        assert isinstance(dataset, Dataset)
        assert len(dataset) == 1

        item = dataset[0]
        assert "prompt" in item
        assert "completion" in item
        assert "text" in item
        assert "instance_id" in item

        # Check that text is prompt + completion
        assert item["text"] == item["prompt"] + item["completion"]

    def test_format_for_sft_filters_long_patches(self, sample_instance):
        """Test that very long patches are filtered out."""
        # Create instance with very long patch
        long_instance = SWEBenchInstance(
            instance_id="long-1",
            repo="test/repo",
            base_commit="abc",
            problem_statement="Test",
            patch="x" * 10000,  # Very long patch
            test_patch="test",
        )

        dataset = format_for_sft(
            [sample_instance, long_instance],
            max_patch_length=8000
        )

        # Only the normal instance should be included
        assert len(dataset) == 1
        assert dataset[0]["instance_id"] == "test__test-1"

    def test_format_for_dpo(self):
        """Test DPO dataset formatting."""
        pairs = [
            PreferencePair(
                prompt="Fix this bug",
                chosen="good patch",
                rejected="bad patch",
                instance_id="test-1",
            ),
            PreferencePair(
                prompt="Fix another bug",
                chosen="good patch 2",
                rejected="bad patch 2",
                instance_id="test-2",
            ),
        ]

        dataset = format_for_dpo(pairs)

        assert isinstance(dataset, Dataset)
        assert len(dataset) == 2

        item = dataset[0]
        assert "prompt" in item
        assert "chosen" in item
        assert "rejected" in item

        assert item["prompt"] == "Fix this bug"
        assert item["chosen"] == "good patch"
        assert item["rejected"] == "bad patch"


class TestPreferencePair:
    def test_preference_pair_creation(self):
        """Test creating a preference pair."""
        pair = PreferencePair(
            prompt="Test prompt",
            chosen="chosen response",
            rejected="rejected response",
            instance_id="test-1",
        )

        assert pair.prompt == "Test prompt"
        assert pair.chosen == "chosen response"
        assert pair.rejected == "rejected response"
        assert pair.instance_id == "test-1"


# Integration tests (require network access)
class TestDataLoading:
    @pytest.mark.slow
    def test_load_swebench_lite(self):
        """Test loading SWE-Bench Lite dataset."""
        from src.data_prep import load_swebench_lite

        # This test requires network access
        instances = load_swebench_lite()

        assert len(instances) > 0
        assert all(isinstance(i, SWEBenchInstance) for i in instances)
        assert all(i.instance_id for i in instances)
        assert all(i.patch for i in instances)

    @pytest.mark.slow
    def test_load_swebench_pro(self):
        """Test loading SWE-Bench Pro dataset."""
        from src.data_prep import load_swebench_pro

        # This test requires network access
        instances = load_swebench_pro()

        assert len(instances) > 0
        assert all(isinstance(i, SWEBenchInstance) for i in instances)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
