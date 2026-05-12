#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"

TORCH_LIB_DIR="$("$PYTHON_BIN" -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
export LD_LIBRARY_PATH="${TORCH_LIB_DIR}:${LD_LIBRARY_PATH:-}"

"$PYTHON_BIN" - <<'PY'
import importlib
import os
import torch

gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
cc = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "N/A"
torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")

print(f"  gpu: {gpu}")
print(f"  cc: {cc}")
print(f"  torch cuda: {torch.version.cuda}")
print(f"  torch lib: {torch_lib}")

importlib.import_module("causal_conv1d_cuda")
print("  causal_conv1d_cuda: ok")
PY
