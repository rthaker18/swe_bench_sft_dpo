#!/usr/bin/env python3
"""
RunPod deployment script for SWE-Patch training.

This script provides utilities for training on RunPod's GPU cloud.
Unlike Modal's serverless approach, RunPod uses persistent instances.

Usage Options:

1. Manual Deployment (Recommended for beginners):
   - Go to runpod.io and create a GPU pod
   - Select a template with PyTorch + CUDA
   - SSH into the pod
   - Clone your repo and run training

2. Programmatic Deployment:
   - Use this script with RunPod's API
   
Prerequisites:
    pip install runpod
    export RUNPOD_API_KEY="your-api-key"
"""

import os
import json
import time
from typing import Dict, Optional
import requests


class RunPodManager:
    """Manage RunPod GPU instances for training."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("RUNPOD_API_KEY")
        if not self.api_key:
            raise ValueError("RUNPOD_API_KEY not set")
        
        self.base_url = "https://api.runpod.io/graphql"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
    
    def _query(self, query: str, variables: Dict = None) -> Dict:
        """Execute GraphQL query."""
        response = requests.post(
            self.base_url,
            headers=self.headers,
            json={"query": query, "variables": variables or {}},
        )
        response.raise_for_status()
        return response.json()
    
    def list_gpu_types(self) -> list:
        """List available GPU types."""
        query = """
        query GpuTypes {
            gpuTypes {
                id
                displayName
                memoryInGb
                secureCloud
            }
        }
        """
        result = self._query(query)
        return result.get("data", {}).get("gpuTypes", [])
    
    def create_pod(
        self,
        name: str,
        gpu_type: str = "NVIDIA RTX A5000",
        cloud_type: str = "SECURE",
        image: str = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel",
        volume_size: int = 50,  # GB
        container_disk: int = 20,  # GB
        env_vars: Dict = None,
    ) -> Dict:
        """
        Create a new GPU pod.
        
        Args:
            name: Pod name
            gpu_type: GPU type (e.g., "NVIDIA RTX A5000", "NVIDIA A100 80GB PCIe")
            cloud_type: "SECURE" or "COMMUNITY"
            image: Docker image
            volume_size: Persistent volume size in GB
            container_disk: Container disk size in GB
            env_vars: Environment variables
            
        Returns:
            Pod info dict
        """
        query = """
        mutation CreatePod($input: PodFindAndDeployOnDemandInput!) {
            podFindAndDeployOnDemand(input: $input) {
                id
                name
                runtime {
                    uptimeInSeconds
                    ports {
                        ip
                        isIpPublic
                        privatePort
                        publicPort
                    }
                }
            }
        }
        """
        
        variables = {
            "input": {
                "name": name,
                "imageName": image,
                "gpuTypeId": gpu_type,
                "cloudType": cloud_type,
                "volumeInGb": volume_size,
                "containerDiskInGb": container_disk,
                "minVcpuCount": 4,
                "minMemoryInGb": 16,
                "gpuCount": 1,
                "startSsh": True,
                "env": [
                    {"key": k, "value": v}
                    for k, v in (env_vars or {}).items()
                ],
            }
        }
        
        result = self._query(query, variables)
        return result.get("data", {}).get("podFindAndDeployOnDemand", {})
    
    def get_pod(self, pod_id: str) -> Dict:
        """Get pod status and info."""
        query = """
        query Pod($podId: String!) {
            pod(input: {podId: $podId}) {
                id
                name
                runtime {
                    uptimeInSeconds
                    ports {
                        ip
                        isIpPublic
                        privatePort
                        publicPort
                    }
                }
                desiredStatus
                machine {
                    podHostId
                }
            }
        }
        """
        result = self._query(query, {"podId": pod_id})
        return result.get("data", {}).get("pod", {})
    
    def terminate_pod(self, pod_id: str) -> bool:
        """Terminate a pod."""
        query = """
        mutation TerminatePod($podId: String!) {
            podTerminate(input: {podId: $podId})
        }
        """
        result = self._query(query, {"podId": pod_id})
        return result.get("data", {}).get("podTerminate", False)


# Training script to run on RunPod
TRAINING_SCRIPT = '''#!/bin/bash
# SWE-Patch Training Script for RunPod

set -e

# Configuration
MODEL_NAME="${MODEL_NAME:-deepseek-ai/deepseek-coder-7b-base-v1.5}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/outputs/sft}"
HF_TOKEN="${HF_TOKEN:-}"
WANDB_API_KEY="${WANDB_API_KEY:-}"
HUB_MODEL_ID="${HUB_MODEL_ID:-}"

echo "================================"
echo "SWE-Patch Training on RunPod"
echo "================================"
echo "Model: $MODEL_NAME"
echo "Output: $OUTPUT_DIR"

# Install dependencies
pip install --upgrade pip
pip install torch transformers datasets accelerate peft trl bitsandbytes
pip install sentencepiece huggingface-hub wandb PyYAML

# Clone training code (or use mounted volume)
if [ ! -d "/workspace/swe_patch_trainer" ]; then
    echo "Cloning training code..."
    git clone https://github.com/YOUR_USERNAME/swe_patch_trainer.git /workspace/swe_patch_trainer
fi

cd /workspace/swe_patch_trainer

# Login to HuggingFace
if [ -n "$HF_TOKEN" ]; then
    echo "Logging in to HuggingFace..."
    huggingface-cli login --token $HF_TOKEN
fi

# Setup wandb
if [ -n "$WANDB_API_KEY" ]; then
    echo "Setting up W&B..."
    wandb login $WANDB_API_KEY
fi

# Prepare data
echo "Preparing training data..."
python -c "
from src.data_prep import load_swebench_pro, format_for_sft, save_dataset
instances = load_swebench_pro()
dataset = format_for_sft(instances)
save_dataset(dataset, './data/sft_train.json')
print(f'Prepared {len(dataset)} training examples')
"

# Run training
echo "Starting SFT training..."
python -m src.sft_trainer \\
    --model "$MODEL_NAME" \\
    --dataset ./data/sft_train.json \\
    --output-dir "$OUTPUT_DIR" \\
    --epochs 3 \\
    ${HUB_MODEL_ID:+--push-to-hub --hub-model-id "$HUB_MODEL_ID"}

echo "================================"
echo "Training complete!"
echo "Output saved to: $OUTPUT_DIR"
echo "================================"
'''


def generate_runpod_dockerfile() -> str:
    """Generate Dockerfile for RunPod training."""
    return '''
# Dockerfile for SWE-Patch Training on RunPod
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy training code
COPY . /workspace/swe_patch_trainer

# Set default command
CMD ["bash"]
'''


def print_manual_instructions():
    """Print instructions for manual RunPod deployment."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║            RunPod Manual Deployment Instructions                  ║
╚══════════════════════════════════════════════════════════════════╝

1. GO TO RUNPOD
   → Visit https://www.runpod.io/console/pods
   → Click "Deploy" to create a new pod

2. SELECT GPU
   Recommended options:
   • RTX A5000 (24GB) - ~$0.30/hr - Good for 7B models
   • A100 40GB - ~$1.89/hr - Best for training
   • A100 80GB - ~$2.69/hr - For larger models

3. SELECT TEMPLATE
   → Choose: "RunPod Pytorch 2.1.0" or similar
   → Or use custom image: runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

4. CONFIGURE
   • Volume Disk: 50GB (for models and data)
   • Container Disk: 20GB
   • Enable SSH access

5. SET ENVIRONMENT VARIABLES
   • HF_TOKEN=your_huggingface_token
   • WANDB_API_KEY=your_wandb_key (optional)
   • HUB_MODEL_ID=your-username/swe-patch-model

6. DEPLOY AND CONNECT
   → Click "Deploy"
   → Wait for pod to start (~2-5 minutes)
   → Use SSH or Web Terminal to connect

7. RUN TRAINING
   Once connected, run:
   
   # Clone repo
   git clone https://github.com/YOUR_USERNAME/swe_patch_trainer.git
   cd swe_patch_trainer
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Prepare data
   python -c "
   from src.data_prep import load_swebench_pro, format_for_sft, save_dataset
   instances = load_swebench_pro()
   dataset = format_for_sft(instances)
   save_dataset(dataset, './data/sft_train.json')
   "
   
   # Run SFT training
   python -m src.sft_trainer \\
       --model deepseek-ai/deepseek-coder-7b-base-v1.5 \\
       --dataset ./data/sft_train.json \\
       --output-dir ./outputs/sft \\
       --epochs 3 \\
       --push-to-hub \\
       --hub-model-id your-username/swe-patch-sft

8. MONITOR TRAINING
   • Check W&B dashboard for metrics
   • Training takes ~8-16 hours for 3 epochs

9. AFTER TRAINING
   • Model is auto-pushed to HuggingFace Hub
   • Download outputs if needed
   • TERMINATE POD to stop charges!

═══════════════════════════════════════════════════════════════════

COST ESTIMATES (approximate):
• RTX A5000: ~$3-5 for full training
• A100 40GB: ~$15-30 for full training

TIP: Use "Savings" pods for ~50% discount (may be interrupted)
""")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RunPod Deployment for SWE-Patch Training")
    parser.add_argument("--create-pod", action="store_true", help="Create a new pod")
    parser.add_argument("--pod-name", type=str, default="swe-patch-training")
    parser.add_argument("--gpu-type", type=str, default="NVIDIA RTX A5000")
    parser.add_argument("--list-gpus", action="store_true", help="List available GPU types")
    parser.add_argument("--terminate", type=str, metavar="POD_ID", help="Terminate a pod")
    parser.add_argument("--status", type=str, metavar="POD_ID", help="Get pod status")
    parser.add_argument("--instructions", action="store_true", help="Print manual instructions")
    parser.add_argument("--print-script", action="store_true", help="Print training script")
    
    args = parser.parse_args()
    
    if args.instructions:
        print_manual_instructions()
    elif args.print_script:
        print(TRAINING_SCRIPT)
    else:
        # Need API key for other operations
        try:
            manager = RunPodManager()
            
            if args.list_gpus:
                gpus = manager.list_gpu_types()
                print("\nAvailable GPU Types:")
                print("-" * 60)
                for gpu in gpus:
                    print(f"  {gpu['displayName']}: {gpu['memoryInGb']}GB")
            
            elif args.create_pod:
                print(f"Creating pod: {args.pod_name}")
                pod = manager.create_pod(
                    name=args.pod_name,
                    gpu_type=args.gpu_type,
                    env_vars={
                        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
                        "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
                    }
                )
                print(f"Pod created: {pod.get('id')}")
                print(f"Name: {pod.get('name')}")
            
            elif args.terminate:
                print(f"Terminating pod: {args.terminate}")
                success = manager.terminate_pod(args.terminate)
                print("Terminated" if success else "Failed to terminate")
            
            elif args.status:
                pod = manager.get_pod(args.status)
                print(json.dumps(pod, indent=2))
            
            else:
                parser.print_help()
                
        except ValueError as e:
            print(f"Error: {e}")
            print("\nFor manual deployment instructions, run:")
            print("  python runpod_train.py --instructions")
