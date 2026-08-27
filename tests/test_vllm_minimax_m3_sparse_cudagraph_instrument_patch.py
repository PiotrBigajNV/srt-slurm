# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import os
import subprocess
from pathlib import Path

PATCHED_SOURCE = '''import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

# One sparse block == one KV page.


@torch.no_grad()
def minimax_m3_sparse_attn_decode(q, max_topk, decode_query_len):
    total_q, num_heads, head_dim = q.shape
    use_pdl = (
        current_platform.is_arch_support_pdl()
        and not torch.cuda.is_current_stream_capturing()
    )
    pdl_launch = {"launch_pdl": True} if use_pdl else {}
    target = max(1, min(max_topk, 256 // max(1, total_q)))
    num_topk_chunks = 1 << (target.bit_length() - 1)
    o_partial = torch.empty(
        1
    )
    _gqa_sparse_decode_kernel[grid](
        USE_PDL=use_pdl,
        **pdl_launch,
    )
    merge_grid = (total_q, num_heads)
    _merge_topk_attn_out_kernel[merge_grid](
        o_partial,
        lse_partial,
        output,
        head_dim,
        o_partial.stride(0),
        o_partial.stride(1),
        o_partial.stride(2),
        o_partial.stride(3),
        lse_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        NUM_TOPK_CHUNKS=num_topk_chunks,
        USE_PDL=use_pdl,
        **pdl_launch,
    )
'''


def _fake_vllm_tree(tmp_path: Path, sparse_source: str) -> tuple[Path, Path, dict[str, str]]:
    vllm_root = tmp_path / "vllm"
    target = vllm_root / "models/minimax_m3/common/ops/sparse_attn.py"
    target.parent.mkdir(parents=True)
    target.write_text(sparse_source)
    (vllm_root / "__init__.py").write_text('__version__ = "test-version"\n')

    base_patch = tmp_path / "base-patch.sh"
    base_patch.write_text("#!/usr/bin/env bash\nset -euo pipefail\necho BASE_PATCH=ok\n")
    base_patch.chmod(0o755)

    script = (
        Path(__file__).parents[1]
        / "configs/patches/patch-vllm-minimax-m3-sparse-cudagraph-instrument.sh"
    )
    env = os.environ | {
        "PYTHONPATH": str(tmp_path),
        "VLLM_MINIMAX_M3_SPARSE_CG_PDL_PATCH_SCRIPT": str(base_patch),
    }
    return target, script, env


def test_sparse_cudagraph_instrument_patch_is_strict_and_idempotent(tmp_path: Path):
    target, script, env = _fake_vllm_tree(tmp_path, PATCHED_SOURCE)

    first = subprocess.run(["bash", str(script)], env=env, text=True, capture_output=True, check=True)
    second = subprocess.run(["bash", str(script)], env=env, text=True, capture_output=True, check=True)
    patched = target.read_text()

    assert "BASE_PATCH=ok" in first.stdout
    assert "VLLM_MINIMAX_M3_SPARSE_CG_DIAG_PATCH=applied" in first.stdout
    assert "VLLM_MINIMAX_M3_SPARSE_CG_DIAG_PATCH=already-applied" in second.stdout
    assert "capturing = torch.cuda.is_current_stream_capturing()" in patched
    assert "event=after_splitk" in patched
    assert "event=after_merge" in patched
    assert patched.count("torch.cuda.current_stream(q.device).synchronize()") == 2
    ast.parse(patched)


def test_sparse_cudagraph_instrument_patch_rejects_unknown_source(tmp_path: Path):
    _, script, env = _fake_vllm_tree(tmp_path, "unexpected source\n")
    result = subprocess.run(["bash", str(script)], env=env, text=True, capture_output=True, check=False)

    assert result.returncode != 0
    assert "Refusing to patch" in result.stderr
