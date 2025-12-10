#!/bin/bash
# Setup script for SWE-Bench evaluation

set -e

echo "=========================================="
echo "Setting up SWE-Bench Evaluation"
echo "=========================================="

# Install SWE-Bench
echo "Installing SWE-Bench..."
pip install swe-bench

# Install Docker if not present (requires sudo)
if ! command -v docker &> /dev/null; then
    echo "Docker not found. Please install Docker manually:"
    echo "https://docs.docker.com/engine/install/"
    exit 1
fi

echo "Docker found: $(docker --version)"

# Check Docker is running
if ! docker info &> /dev/null; then
    echo "Docker daemon is not running. Please start Docker."
    exit 1
fi

# Create necessary directories
mkdir -p evaluation_results
mkdir -p evaluation_logs
mkdir -p testbed

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Generate predictions:"
echo "   python scripts/evaluate_swebench.py --adapter-model <your-model>"
echo ""
echo "2. Run evaluation (requires Docker):"
echo "   python -m swebench.harness.run_evaluation \\"
echo "       --predictions_path evaluation_results/predictions_*.jsonl \\"
echo "       --swe_bench_tasks pro \\"
echo "       --log_dir ./evaluation_logs \\"
echo "       --testbed ./testbed \\"
echo "       --num_workers 4"
echo ""
