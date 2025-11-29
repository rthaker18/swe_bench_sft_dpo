"""
Docker-based evaluation runner for SWE-Bench instances.

This module manages Docker containers to run SWE-Bench evaluations
in isolated, reproducible environments.
"""

import os
import json
import time
import tempfile
import subprocess
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

import docker
from docker.models.containers import Container
from tqdm import tqdm


@dataclass
class EvaluationResult:
    """Result of evaluating a single instance."""
    instance_id: str
    passed: bool
    tests_passed: int = 0
    tests_failed: int = 0
    tests_error: int = 0
    patch_applied: bool = True
    error_message: Optional[str] = None
    execution_time: float = 0.0
    logs: str = ""


@dataclass
class SWEBenchInstance:
    """Minimal instance data needed for evaluation."""
    instance_id: str
    repo: str
    base_commit: str
    test_patch: str
    patch: str  # Gold patch (for reference)


class DockerRunner:
    """
    Manages Docker containers for SWE-Bench evaluation.
    
    Uses pre-built images from the SWE-Bench Pro Docker Hub:
    https://hub.docker.com/r/jefzda/sweap-images
    """
    
    # Docker Hub prefix for SWE-Bench Pro images
    IMAGE_PREFIX = "jefzda/sweap-images"
    
    def __init__(
        self,
        timeout: int = 300,  # 5 minutes per instance
        memory_limit: str = "8g",
        cpu_limit: float = 2.0,
    ):
        """
        Initialize Docker runner.
        
        Args:
            timeout: Maximum execution time per instance (seconds)
            memory_limit: Container memory limit
            cpu_limit: Number of CPU cores
        """
        self.client = docker.from_env()
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        
        # Verify Docker is available
        try:
            self.client.ping()
            print("Docker connection established")
        except docker.errors.DockerException as e:
            raise RuntimeError(f"Failed to connect to Docker: {e}")
    
    def get_image_name(self, instance: SWEBenchInstance) -> str:
        """
        Get the Docker image name for an instance.
        
        SWE-Bench Pro format:
        jefzda/sweap-images:{repo_base}.{repo_name}-{repo_base}__{repo_name}-{hash}
        """
        # Parse repo (e.g., "django/django" -> "django", "django")
        repo_parts = instance.repo.split("/")
        repo_base = repo_parts[0]
        repo_name = repo_parts[1] if len(repo_parts) > 1 else repo_parts[0]
        
        # Format: base.name-base__name-commit_hash
        image_tag = f"{repo_base}.{repo_name}-{repo_base}__{repo_name}-{instance.base_commit}"
        
        return f"{self.IMAGE_PREFIX}:{image_tag}"
    
    def pull_image(self, image_name: str) -> bool:
        """Pull Docker image if not available locally."""
        try:
            self.client.images.get(image_name)
            return True
        except docker.errors.ImageNotFound:
            print(f"Pulling image: {image_name}")
            try:
                self.client.images.pull(image_name)
                return True
            except docker.errors.APIError as e:
                print(f"Failed to pull image {image_name}: {e}")
                return False
    
    def apply_patch(self, container: Container, patch: str) -> Tuple[bool, str]:
        """
        Apply a patch inside the container.
        
        Args:
            container: Running container
            patch: Unified diff patch string
            
        Returns:
            (success, error_message)
        """
        # Write patch to temp file in container
        patch_cmd = f"""
cat << 'PATCH_EOF' > /tmp/model_patch.diff
{patch}
PATCH_EOF
"""
        exit_code, output = container.exec_run(
            ["bash", "-c", patch_cmd],
            workdir="/testbed"
        )
        
        if exit_code != 0:
            return False, f"Failed to write patch: {output.decode()}"
        
        # Apply the patch
        apply_cmd = "cd /testbed && git apply /tmp/model_patch.diff"
        exit_code, output = container.exec_run(
            ["bash", "-c", apply_cmd],
            workdir="/testbed"
        )
        
        if exit_code != 0:
            return False, f"Failed to apply patch: {output.decode()}"
        
        return True, ""
    
    def apply_test_patch(self, container: Container, test_patch: str) -> Tuple[bool, str]:
        """Apply the test patch to add/update tests."""
        if not test_patch:
            return True, ""
        
        # Write test patch
        patch_cmd = f"""
cat << 'PATCH_EOF' > /tmp/test_patch.diff
{test_patch}
PATCH_EOF
"""
        exit_code, output = container.exec_run(
            ["bash", "-c", patch_cmd],
            workdir="/testbed"
        )
        
        if exit_code != 0:
            return False, f"Failed to write test patch: {output.decode()}"
        
        # Apply test patch
        apply_cmd = "cd /testbed && git apply /tmp/test_patch.diff"
        exit_code, output = container.exec_run(
            ["bash", "-c", apply_cmd],
            workdir="/testbed"
        )
        
        if exit_code != 0:
            return False, f"Failed to apply test patch: {output.decode()}"
        
        return True, ""
    
    def run_tests(self, container: Container, instance: SWEBenchInstance) -> Tuple[int, int, int, str]:
        """
        Run tests in the container.
        
        Returns:
            (passed, failed, errors, output)
        """
        # Default test command - this varies by repo
        # SWE-Bench instances usually have a specific test command
        test_cmd = "cd /testbed && python -m pytest -xvs 2>&1"
        
        try:
            exit_code, output = container.exec_run(
                ["bash", "-c", test_cmd],
                workdir="/testbed",
                timeout=self.timeout - 60,  # Leave some buffer
            )
            output_str = output.decode("utf-8", errors="replace")
        except Exception as e:
            return 0, 0, 1, str(e)
        
        # Parse pytest output for pass/fail counts
        passed = 0
        failed = 0
        errors = 0
        
        for line in output_str.split("\n"):
            line_lower = line.lower()
            if "passed" in line_lower:
                try:
                    # Parse "X passed" format
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if p == "passed" and i > 0:
                            passed = int(parts[i-1])
                            break
                except:
                    pass
            if "failed" in line_lower:
                try:
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if p == "failed" and i > 0:
                            failed = int(parts[i-1])
                            break
                except:
                    pass
            if "error" in line_lower:
                try:
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if p == "error" or p == "errors" and i > 0:
                            errors = int(parts[i-1])
                            break
                except:
                    pass
        
        return passed, failed, errors, output_str
    
    def evaluate_instance(
        self,
        instance: SWEBenchInstance,
        model_patch: str,
    ) -> EvaluationResult:
        """
        Evaluate a single instance with a model-generated patch.
        
        Args:
            instance: SWE-Bench instance
            model_patch: Model-generated patch to evaluate
            
        Returns:
            EvaluationResult
        """
        start_time = time.time()
        
        # Get image name
        image_name = self.get_image_name(instance)
        
        # Pull image if needed
        if not self.pull_image(image_name):
            return EvaluationResult(
                instance_id=instance.instance_id,
                passed=False,
                error_message=f"Failed to pull image: {image_name}",
                execution_time=time.time() - start_time,
            )
        
        container = None
        try:
            # Start container
            container = self.client.containers.run(
                image_name,
                detach=True,
                mem_limit=self.memory_limit,
                # Note: nano_cpus expects integer nanoseconds
                nano_cpus=int(self.cpu_limit * 1e9),
                remove=False,  # Keep for debugging
            )
            
            # Wait for container to be ready
            time.sleep(2)
            
            # Apply model patch
            success, error = self.apply_patch(container, model_patch)
            if not success:
                return EvaluationResult(
                    instance_id=instance.instance_id,
                    passed=False,
                    patch_applied=False,
                    error_message=error,
                    execution_time=time.time() - start_time,
                )
            
            # Apply test patch
            success, error = self.apply_test_patch(container, instance.test_patch)
            if not success:
                return EvaluationResult(
                    instance_id=instance.instance_id,
                    passed=False,
                    error_message=f"Test patch failed: {error}",
                    execution_time=time.time() - start_time,
                )
            
            # Run tests
            passed, failed, errors, logs = self.run_tests(container, instance)
            
            # Determine overall pass/fail
            # Pass if no failures and no errors
            overall_passed = failed == 0 and errors == 0 and passed > 0
            
            return EvaluationResult(
                instance_id=instance.instance_id,
                passed=overall_passed,
                tests_passed=passed,
                tests_failed=failed,
                tests_error=errors,
                patch_applied=True,
                logs=logs[-5000:] if len(logs) > 5000 else logs,  # Truncate logs
                execution_time=time.time() - start_time,
            )
            
        except Exception as e:
            return EvaluationResult(
                instance_id=instance.instance_id,
                passed=False,
                error_message=str(e),
                execution_time=time.time() - start_time,
            )
        finally:
            # Cleanup container
            if container:
                try:
                    container.stop(timeout=5)
                    container.remove()
                except:
                    pass
    
    def evaluate_batch(
        self,
        instances: List[SWEBenchInstance],
        patches: Dict[str, str],  # instance_id -> patch
        max_workers: int = 4,
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple instances in parallel.
        
        Args:
            instances: List of instances to evaluate
            patches: Dict mapping instance_id to patch
            max_workers: Number of parallel evaluations
            
        Returns:
            List of EvaluationResult
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {}
            for instance in instances:
                patch = patches.get(instance.instance_id)
                if patch:
                    future = executor.submit(
                        self.evaluate_instance, instance, patch
                    )
                    futures[future] = instance.instance_id
            
            # Collect results
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Evaluating"
            ):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    instance_id = futures[future]
                    results.append(EvaluationResult(
                        instance_id=instance_id,
                        passed=False,
                        error_message=str(e),
                    ))
        
        return results


class SimplifiedEvaluator:
    """
    Simplified evaluator that doesn't require Docker.
    
    Uses subprocess to run git and tests locally.
    Good for testing and development.
    """
    
    def __init__(self, work_dir: str = "/tmp/swe_eval"):
        self.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)
    
    def clone_repo(self, repo: str, commit: str, target_dir: str) -> bool:
        """Clone repo and checkout specific commit."""
        try:
            # Clone
            subprocess.run(
                ["git", "clone", f"https://github.com/{repo}.git", target_dir],
                check=True,
                capture_output=True,
            )
            # Checkout
            subprocess.run(
                ["git", "checkout", commit],
                cwd=target_dir,
                check=True,
                capture_output=True,
            )
            return True
        except subprocess.CalledProcessError:
            return False
    
    def apply_patch(self, patch: str, work_dir: str) -> bool:
        """Apply patch using git apply."""
        patch_file = os.path.join(work_dir, "patch.diff")
        with open(patch_file, "w") as f:
            f.write(patch)
        
        try:
            subprocess.run(
                ["git", "apply", patch_file],
                cwd=work_dir,
                check=True,
                capture_output=True,
            )
            return True
        except subprocess.CalledProcessError:
            return False


# Example usage
if __name__ == "__main__":
    # Test with a simple instance
    runner = DockerRunner()
    
    # Example instance (you'd load this from dataset)
    instance = SWEBenchInstance(
        instance_id="test__test-123",
        repo="test/test",
        base_commit="abc123",
        test_patch="",
        patch="",
    )
    
    # Example patch
    model_patch = """
diff --git a/file.py b/file.py
--- a/file.py
+++ b/file.py
@@ -1,1 +1,1 @@
-old line
+new line
"""
    
    print("Testing Docker runner...")
    # result = runner.evaluate_instance(instance, model_patch)
    # print(f"Result: {result}")
