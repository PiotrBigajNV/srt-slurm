#!/usr/bin/env bash
set -euo pipefail

BASE_PATCH="${VLLM_MINIMAX_M3_CUTLASS_BATCH1_BASE_PATCH_SCRIPT:-/configs/patches/patch-vllm-minimax-m3-sparse-cudagraph-no-pdl.sh}"
PROBE_SCRIPT="${VLLM_MINIMAX_M3_CUTLASS_BATCH1_PROBE_SCRIPT:-/configs/patches/probe-vllm-minimax-m3-cutlass-batch1.py}"

bash "${BASE_PATCH}"

if [[ ! -f "${PROBE_SCRIPT}" ]]; then
    echo "Missing MiniMax-M3 CUTLASS batch-1 probe: ${PROBE_SCRIPT}" >&2
    exit 1
fi

# GPU workers must prove eager correctness plus graph capture/replay before the
# threshold is changed. CPU-only frontend containers emit an explicit skip;
# they do not claim that the probe passed.
python3 "${PROBE_SCRIPT}"

VLLM_ROOT="$(python3 - <<'PY'
from pathlib import Path

import vllm

print(Path(vllm.__file__).resolve().parent)
PY
)"
TARGET="${VLLM_ROOT}/models/minimax_m3/nvidia/msa_cutlass_sparse_decode.py"

python3 - "${TARGET}" <<'PY'
from pathlib import Path
import sys

target = Path(sys.argv[1])
source = target.read_text()

old = "_MIN_CUTLASS_BATCH_SIZE = 16\n"
new = "_MIN_CUTLASS_BATCH_SIZE = 1\n"

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
        f"CUTLASS minimum-batch declaration (old={old_count}, new={new_count})"
    )

verified = target.read_text()
if old in verified or verified.count(new) != 1:
    raise SystemExit(f"MiniMax-M3 CUTLASS batch-1 patch verification failed for {target}")

compile(verified, str(target), "exec")
print(f"VLLM_MINIMAX_M3_CUTLASS_BATCH1_PATCH={state}")
print(f"VLLM_MINIMAX_M3_CUTLASS_BATCH1_PATCH_TARGET={target}")
print("VLLM_MINIMAX_M3_CUTLASS_BATCH1_PATCH_THRESHOLD=1")
PY

python3 - <<'PY'
import vllm

print(f"VLLM_MINIMAX_M3_CUTLASS_BATCH1_PATCH_VLLM_VERSION={vllm.__version__}")
PY
