#!/usr/bin/env bash
set -euo pipefail

BASE_PATCH="${VLLM_MINIMAX_M3_SPARSE_CG_INSTRUMENT_PATCH_SCRIPT:-/configs/patches/patch-vllm-minimax-m3-sparse-cudagraph-instrument.sh}"
bash "${BASE_PATCH}"

VLLM_ROOT="$(python3 - <<'PY'
from pathlib import Path

import vllm

print(Path(vllm.__file__).resolve().parent)
PY
)"
TARGET="${VLLM_ROOT}/models/minimax_m3/common/ops/sparse_attn.py"

python3 - "${TARGET}" <<'PY'
from pathlib import Path
import sys

target = Path(sys.argv[1])
source = target.read_text()

old = '''    capturing = torch.cuda.is_current_stream_capturing()
    use_pdl = current_platform.is_arch_support_pdl() and not capturing
'''
new = '''    capturing = torch.cuda.is_current_stream_capturing()
    # FULL CUDA-graph profiling executes a warmup before the stream begins
    # capturing. On SM107 that warmup faults in the PDL split-K sparse kernel,
    # so capture-state gating alone cannot protect the FULL descriptor. Keep
    # CUDA graphs enabled, but use the existing non-PDL Triton variants for
    # warmup, capture, and replay preparation in this controlled diagnostic.
    use_pdl = False
'''

old_count = source.count(old)
new_count = source.count(new)
if old_count == 1 and new_count == 0:
    source = source.replace(old, new, 1)
    target.write_text(source)
    state = "applied"
elif old_count == 0 and new_count == 1:
    state = "already-applied"
else:
    raise SystemExit(
        f"Refusing to patch {target}: expected exactly one instrumented "
        f"MiniMax-M3 PDL-selection block (old={old_count}, new={new_count})"
    )

verified = target.read_text()
if old in verified or verified.count(new) != 1:
    raise SystemExit(f"MiniMax-M3 sparse no-PDL patch verification failed for {target}")

compile(verified, str(target), "exec")
print(f"VLLM_MINIMAX_M3_SPARSE_NO_PDL_PATCH={state}")
print(f"VLLM_MINIMAX_M3_SPARSE_NO_PDL_PATCH_TARGET={target}")
PY

python3 - <<'PY'
import vllm

print(f"VLLM_MINIMAX_M3_SPARSE_NO_PDL_PATCH_VLLM_VERSION={vllm.__version__}")
PY
