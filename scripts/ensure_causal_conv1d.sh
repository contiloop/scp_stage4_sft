#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"

TORCH_LIB_DIR="$("$PYTHON_BIN" -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
export LD_LIBRARY_PATH="${TORCH_LIB_DIR}:${LD_LIBRARY_PATH:-}"

CC="$("$PYTHON_BIN" -c 'import torch; print(".".join(map(str, torch.cuda.get_device_capability(0))) if torch.cuda.is_available() else "cpu")')"
CURRENT_VER="$("$PYTHON_BIN" - <<'PY'
import importlib.metadata as metadata

try:
    print(metadata.version("causal-conv1d"))
except Exception:
    print("missing")
PY
)"

echo "  causal_conv1d check: cc=${CC}, current=${CURRENT_VER}"

kernel_smoke_test() {
  "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import torch
from causal_conv1d import causal_conv1d_fn

if not torch.cuda.is_available():
    raise SystemExit(0)

x = torch.randn(1, 32, 64, device="cuda", dtype=torch.float16)
w = torch.randn(32, 4, device="cuda", dtype=torch.float16)
_ = causal_conv1d_fn(x, w)
torch.cuda.synchronize()
PY
}

ensure_installed() {
  "$PYTHON_BIN" -c "import causal_conv1d" 2>/dev/null || "$PYTHON_BIN" -m pip install causal-conv1d
}

ensure_installed
if kernel_smoke_test; then
  echo "  causal_conv1d kernel smoke test: ok (skip rebuild)"
  exit 0
fi

if [[ "${CC}" == "12.0" ]]; then
  echo "  Blackwell detected and kernel test failed -> rebuild causal-conv1d==1.6.1 from source"
  "$PYTHON_BIN" -m pip uninstall -y causal-conv1d >/dev/null 2>&1 || true
  CAUSAL_CONV1D_FORCE_BUILD=TRUE TORCH_CUDA_ARCH_LIST=12.0 \
    "$PYTHON_BIN" -m pip install -v --no-build-isolation --no-binary :all: causal-conv1d==1.6.1
else
  echo "  non-Blackwell kernel test failed -> reinstall causal-conv1d"
  "$PYTHON_BIN" -m pip uninstall -y causal-conv1d >/dev/null 2>&1 || true
  "$PYTHON_BIN" -m pip install causal-conv1d
fi

if kernel_smoke_test; then
  echo "  causal_conv1d kernel smoke test: ok (after install)"
else
  echo "  [ERROR] causal_conv1d kernel smoke test failed after install"
  exit 1
fi
