# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import os
import subprocess
from pathlib import Path

INSTRUMENTED_SOURCE = '''import torch

from vllm.platforms import current_platform


def minimax_m3_sparse_attn_decode(q):
    capturing = torch.cuda.is_current_stream_capturing()
    use_pdl = current_platform.is_arch_support_pdl() and not capturing
    print(f"use_pdl={int(use_pdl)}")
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
        / "configs/patches/patch-vllm-minimax-m3-sparse-cudagraph-no-pdl.sh"
    )
    env = os.environ | {
        "PYTHONPATH": str(tmp_path),
        "VLLM_MINIMAX_M3_SPARSE_CG_INSTRUMENT_PATCH_SCRIPT": str(base_patch),
    }
    return target, script, env


def test_sparse_cudagraph_no_pdl_patch_is_strict_and_idempotent(tmp_path: Path):
    target, script, env = _fake_vllm_tree(tmp_path, INSTRUMENTED_SOURCE)

    first = subprocess.run(["bash", str(script)], env=env, text=True, capture_output=True, check=True)
    second = subprocess.run(["bash", str(script)], env=env, text=True, capture_output=True, check=True)
    patched = target.read_text()

    assert "BASE_PATCH=ok" in first.stdout
    assert "VLLM_MINIMAX_M3_SPARSE_NO_PDL_PATCH=applied" in first.stdout
    assert "VLLM_MINIMAX_M3_SPARSE_NO_PDL_PATCH=already-applied" in second.stdout
    assert "capturing = torch.cuda.is_current_stream_capturing()" in patched
    assert "use_pdl = False" in patched
    assert "current_platform.is_arch_support_pdl() and not capturing" not in patched
    ast.parse(patched)


def test_sparse_cudagraph_no_pdl_patch_rejects_unknown_source(tmp_path: Path):
    _, script, env = _fake_vllm_tree(tmp_path, "unexpected source\n")
    result = subprocess.run(["bash", str(script)], env=env, text=True, capture_output=True, check=False)

    assert result.returncode != 0
    assert "Refusing to patch" in result.stderr
