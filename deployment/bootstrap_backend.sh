#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
sub=SwiftTransformer/src/csrc/kernel/xformers
if [ ! -d "$sub/.git" ]; then
  if [ -n "$(ls -A "$sub" 2>/dev/null)" ]; then echo "Nonempty xformers without git; refusing overwrite" >&2; exit 1; fi
  git clone https://github.com/facebookresearch/xformers.git "$sub"
fi
git -C "$sub" checkout --detach ad986981b141a218bf07bf968e920051ff2c7b41
git -C "$sub" submodule update --init --recursive
cmake -S SwiftTransformer -B SwiftTransformer/build -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build SwiftTransformer/build -j "${BUILD_JOBS:-4}"
python -m pip install -e .
