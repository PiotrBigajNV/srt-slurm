#!/usr/bin/env bash
set -euo pipefail

BASE_PATCH="${VLLM_MOONCAKE_MIXED_REGION_PATCH_SCRIPT:-/configs/patches/patch-vllm-mooncake-mixed-regions.sh}"
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

old = '''    use_pdl = current_platform.is_arch_support_pdl()
    # `launch_pdl` is a Triton runtime kwarg only some backends accept (CUDA
    # SM9+); this ROCm Triton rejects it even when False ("Keyword argument
    # launch_pdl was specified but unrecognised"). Only pass it when PDL is
    # actually supported -- on ROCm use_pdl is always False, so it's omitted.
'''
new = '''    # Triton's Programmatic Dependent Launch path is valid for eager execution,
    # but the SM107/CUDA 13.5 stack faults when this split-K/merge pair is
    # recorded into a FULL CUDA graph. The Python wrapper runs while the graph
    # is captured, so bake the non-PDL kernel variants into that graph. Replay
    # uses those captured variants; eager and PIECEWISE-break execution retain
    # PDL because their streams are not capturing.
    use_pdl = (
        current_platform.is_arch_support_pdl()
        and not torch.cuda.is_current_stream_capturing()
    )
    # `launch_pdl` is a Triton runtime kwarg only some backends accept (CUDA
    # SM9+); this ROCm Triton rejects it even when False ("Keyword argument
    # launch_pdl was specified but unrecognised"). Only pass it when PDL is
    # actually supported and the current stream is not being captured.
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
        f"Refusing to patch {target}: expected exactly one known MiniMax-M3 "
        f"sparse-decode PDL block (old={old_count}, new={new_count})"
    )

verified = target.read_text()
if old in verified or verified.count(new) != 1:
    raise SystemExit(f"MiniMax-M3 sparse CUDA-graph PDL patch verification failed for {target}")

compile(verified, str(target), "exec")
print(f"VLLM_MINIMAX_M3_SPARSE_CG_PDL_PATCH={state}")
print(f"VLLM_MINIMAX_M3_SPARSE_CG_PDL_PATCH_TARGET={target}")
PY

python3 - <<'PY'
import vllm

print(f"VLLM_MINIMAX_M3_SPARSE_CG_PDL_PATCH_VLLM_VERSION={vllm.__version__}")
PY
