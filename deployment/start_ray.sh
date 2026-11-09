#!/usr/bin/env bash
set -euo pipefail
# Call on each node after inspecting the physical GPUs and reserving ports.
: "${GPU_IDS:?Set exactly two allocated physical GPU IDs}"
: "${NODE_IP:?Set this node address}"
: "${ROLE:?Set prefill or decode}"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
python - <<'PY'
import os, subprocess
ids = os.environ['GPU_IDS'].split(',')
if len(ids)!=2 or len(set(ids))!=2: raise SystemExit('Exactly two distinct GPUs required')
expected={'prefill':'A6000','decode':'A5000'}[os.environ['ROLE']]
for gpu in ids:
    name=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=name','--format=csv,noheader'],text=True)
    if expected not in name: raise SystemExit('Wrong GPU model')
    busy=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-compute-apps=pid','--format=csv,noheader'],text=True)
    if busy.strip(): raise SystemExit('Allocated GPU is already occupied')
PY
if [ "$ROLE" = prefill ]; then
  : "${RAY_PORT:?Set a free Ray head port}"
  ray start --head --node-ip-address="$NODE_IP" --port="$RAY_PORT" --num-gpus=2 --include-dashboard=false
else
  : "${RAY_HEAD:?Set head-address:port}"
  ray start --address="$RAY_HEAD" --node-ip-address="$NODE_IP" --num-gpus=2
fi
