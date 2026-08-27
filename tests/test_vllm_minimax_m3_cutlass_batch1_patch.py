# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import os
import subprocess
from pathlib import Path

CUTLASS_SOURCE = '''_MIN_CUTLASS_BATCH_SIZE = 16


def should_prepare_decode_metadata(batch_size):
    return batch_size >= _MIN_CUTLASS_BATCH_SIZE
'''


def _fake_vllm_tree(
    tmp_path: Path,
    cutlass_source: str,
    *,
    probe_exit: int = 0,
) -> tuple[Path, Path, dict[str, str]]:
    vllm_root = tmp_path / "vllm"
    target = vllm_root / "models/minimax_m3/nvidia/msa_cutlass_sparse_decode.py"
    target.parent.mkdir(parents=True)
    target.write_text(cutlass_source)
    (vllm_root / "__init__.py").write_text('__version__ = "test-version"\n')

    base_patch = tmp_path / "base-patch.sh"
    base_patch.write_text("#!/usr/bin/env bash\nset -euo pipefail\necho BASE_PATCH=ok\n")
    base_patch.chmod(0o755)

    probe = tmp_path / "probe.py"
    probe.write_text(
        "import sys\n"
        f'print("VLLM_MINIMAX_M3_CUTLASS_BATCH1_PROBE=test-{probe_exit}")\n'
        f"sys.exit({probe_exit})\n"
    )

    script = (
        Path(__file__).parents[1]
        / "configs/patches/patch-vllm-minimax-m3-cutlass-batch1.sh"
    )
    env = os.environ | {
        "PYTHONPATH": str(tmp_path),
        "VLLM_MINIMAX_M3_CUTLASS_BATCH1_BASE_PATCH_SCRIPT": str(base_patch),
        "VLLM_MINIMAX_M3_CUTLASS_BATCH1_PROBE_SCRIPT": str(probe),
    }
    return target, script, env


def test_cutlass_batch1_patch_is_probe_gated_strict_and_idempotent(tmp_path: Path):
    target, script, env = _fake_vllm_tree(tmp_path, CUTLASS_SOURCE)

    first = subprocess.run(["bash", str(script)], env=env, text=True, capture_output=True, check=True)
    second = subprocess.run(["bash", str(script)], env=env, text=True, capture_output=True, check=True)
    patched = target.read_text()

    assert "BASE_PATCH=ok" in first.stdout
    assert "VLLM_MINIMAX_M3_CUTLASS_BATCH1_PROBE=test-0" in first.stdout
    assert "VLLM_MINIMAX_M3_CUTLASS_BATCH1_PATCH=applied" in first.stdout
    assert "VLLM_MINIMAX_M3_CUTLASS_BATCH1_PATCH=already-applied" in second.stdout
    assert "_MIN_CUTLASS_BATCH_SIZE = 1" in patched
    assert "_MIN_CUTLASS_BATCH_SIZE = 16" not in patched
    ast.parse(patched)


def test_cutlass_batch1_patch_does_not_patch_when_probe_fails(tmp_path: Path):
    target, script, env = _fake_vllm_tree(tmp_path, CUTLASS_SOURCE, probe_exit=23)

    result = subprocess.run(["bash", str(script)], env=env, text=True, capture_output=True, check=False)

    assert result.returncode == 23
    assert target.read_text() == CUTLASS_SOURCE
    assert "VLLM_MINIMAX_M3_CUTLASS_BATCH1_PATCH=" not in result.stdout


def test_cutlass_batch1_patch_rejects_unknown_source(tmp_path: Path):
    _, script, env = _fake_vllm_tree(tmp_path, "unexpected source\n")
    result = subprocess.run(["bash", str(script)], env=env, text=True, capture_output=True, check=False)

    assert result.returncode != 0
    assert "Refusing to patch" in result.stderr


def test_cutlass_batch1_probe_parses() -> None:
    probe = (
        Path(__file__).parents[1]
        / "configs/patches/probe-vllm-minimax-m3-cutlass-batch1.py"
    )
    ast.parse(probe.read_text())
