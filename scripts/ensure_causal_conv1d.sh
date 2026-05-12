#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"
CAUSAL_CONV1D_VERSION="${CAUSAL_CONV1D_VERSION:-1.6.2.post1}"

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
  "$PYTHON_BIN" -m pip install -q --upgrade "setuptools>=70.1.0" wheel packaging ninja numpy

  if "$PYTHON_BIN" -c "import causal_conv1d" >/dev/null 2>&1; then
    return 0
  fi

  # Avoid build-isolation pulling a different torch (e.g. 2.11+cu130) and
  # forcing source builds with CUDA mismatch. Try release wheels that match the
  # current torch runtime first.
  wheel_urls="$("$PYTHON_BIN" - <<'PY'
import os, sys, torch

ver = os.environ.get("CAUSAL_CONV1D_VERSION", "1.6.2.post1")
torch_ver = ".".join(torch.__version__.split("+")[0].split(".")[:2])  # 2.10
cuda_ver = str(torch.version.cuda or "12.8")
cuda_major = cuda_ver.split(".")[0]  # 12/13
cp = f"cp{sys.version_info.major}{sys.version_info.minor}"
base = f"https://github.com/Dao-AILab/causal-conv1d/releases/download/v{ver}"
for abi in ("TRUE", "FALSE"):
    print(
        f"{base}/causal_conv1d-{ver}+cu{cuda_major}torch{torch_ver}cxx11abi{abi}-{cp}-{cp}-linux_x86_64.whl"
    )
PY
)"

  while IFS= read -r url; do
    [ -n "$url" ] || continue
    echo "  trying prebuilt wheel: $url"
    if "$PYTHON_BIN" -m pip install --no-deps --no-build-isolation "$url"; then
      return 0
    fi
  done <<< "$wheel_urls"

  # Final fallback.
  "$PYTHON_BIN" -m pip install --no-deps --no-build-isolation "causal-conv1d==${CAUSAL_CONV1D_VERSION}"
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
