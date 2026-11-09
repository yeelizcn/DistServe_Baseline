#!/usr/bin/env bash
set -euo pipefail
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,driver_version --format=csv
nvidia-smi topo -m
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
uname -a
lscpu
free -h
df -h
ip -brief address
python -c 'import torch, ray, transformers; print(torch.__version__, torch.version.cuda, torch.cuda.nccl.version(), ray.__version__, transformers.__version__)'
