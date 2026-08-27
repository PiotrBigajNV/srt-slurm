#!/usr/bin/env bash
set -euo pipefail

BASE_PATCH="${VLLM_MINIMAX_M3_SPARSE_CG_PDL_PATCH_SCRIPT:-/configs/patches/patch-vllm-minimax-m3-sparse-cudagraph-pdl.sh}"
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

old_state = '''from vllm.triton_utils import tl, triton

# One sparse block == one KV page.
'''
new_state = '''from vllm.triton_utils import tl, triton

_MINIMAX_M3_SPARSE_CG_DIAG_COUNTS: dict[tuple[object, ...], int] = {}

# One sparse block == one KV page.
'''

old_pdl = '''    use_pdl = (
        current_platform.is_arch_support_pdl()
        and not torch.cuda.is_current_stream_capturing()
    )
'''
new_pdl = '''    capturing = torch.cuda.is_current_stream_capturing()
    use_pdl = current_platform.is_arch_support_pdl() and not capturing
'''

old_diag = '''    num_topk_chunks = 1 << (target.bit_length() - 1)
    o_partial = torch.empty(
'''
new_diag = '''    num_topk_chunks = 1 << (target.bit_length() - 1)
    diag_key = (
        capturing,
        use_pdl,
        total_q,
        max_topk,
        num_topk_chunks,
        decode_query_len,
        str(q.device),
    )
    diag_index = _MINIMAX_M3_SPARSE_CG_DIAG_COUNTS.get(diag_key, 0)
    _MINIMAX_M3_SPARSE_CG_DIAG_COUNTS[diag_key] = diag_index + 1
    diag_active = diag_index < 2
    if diag_active:
        print(
            "VLLM_MINIMAX_M3_SPARSE_CG_DIAG "
            f"event=begin call={diag_index + 1} capturing={int(capturing)} "
            f"use_pdl={int(use_pdl)} total_q={total_q} max_topk={max_topk} "
            f"chunks={num_topk_chunks} decode_query_len={decode_query_len} "
            f"device={q.device}",
            flush=True,
        )
    o_partial = torch.empty(
'''

old_split_sync = '''        **pdl_launch,
    )
    merge_grid = (total_q, num_heads)
'''
new_split_sync = '''        **pdl_launch,
    )
    if diag_active and not capturing:
        torch.cuda.current_stream(q.device).synchronize()
        print(
            "VLLM_MINIMAX_M3_SPARSE_CG_DIAG "
            f"event=after_splitk call={diag_index + 1} capturing=0 "
            f"use_pdl={int(use_pdl)} total_q={total_q} chunks={num_topk_chunks}",
            flush=True,
        )
    merge_grid = (total_q, num_heads)
'''

old_merge_sync = '''        output.stride(2),
        NUM_TOPK_CHUNKS=num_topk_chunks,
        USE_PDL=use_pdl,
        **pdl_launch,
    )
'''
new_merge_sync = '''        output.stride(2),
        NUM_TOPK_CHUNKS=num_topk_chunks,
        USE_PDL=use_pdl,
        # The diagnostic synchronizes immediately after this launch when safe.
        **pdl_launch,
    )
    if diag_active and not capturing:
        torch.cuda.current_stream(q.device).synchronize()
        print(
            "VLLM_MINIMAX_M3_SPARSE_CG_DIAG "
            f"event=after_merge call={diag_index + 1} capturing=0 "
            f"use_pdl={int(use_pdl)} total_q={total_q} chunks={num_topk_chunks}",
            flush=True,
        )
'''

patches = (
    ("state", old_state, new_state),
    ("capture-state", old_pdl, new_pdl),
    ("diagnostic-header", old_diag, new_diag),
    ("split-k-sync", old_split_sync, new_split_sync),
    ("merge-sync", old_merge_sync, new_merge_sync),
)

states: list[str] = []
for name, old, new in patches:
    old_count = source.count(old)
    new_count = source.count(new)
    if old_count == 1 and new_count == 0:
        source = source.replace(old, new, 1)
        states.append("applied")
    elif old_count == 0 and new_count == 1:
        states.append("already-applied")
    else:
        raise SystemExit(
            f"Refusing to patch {target}: expected one known {name} block "
            f"(old={old_count}, new={new_count})"
        )

target.write_text(source)
verified = target.read_text()
for name, old, new in patches:
    if old in verified or verified.count(new) != 1:
        raise SystemExit(f"MiniMax-M3 sparse CUDA-graph diagnostic verification failed: {name}")

compile(verified, str(target), "exec")
state = "applied" if "applied" in states else "already-applied"
print(f"VLLM_MINIMAX_M3_SPARSE_CG_DIAG_PATCH={state}")
print(f"VLLM_MINIMAX_M3_SPARSE_CG_DIAG_PATCH_TARGET={target}")
PY

python3 - <<'PY'
import vllm

print(f"VLLM_MINIMAX_M3_SPARSE_CG_DIAG_PATCH_VLLM_VERSION={vllm.__version__}")
PY
