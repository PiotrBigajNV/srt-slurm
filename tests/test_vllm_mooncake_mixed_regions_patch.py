# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from test_vllm_multiconnector_prometheus_patch import OLD_BLOCK

OLD_MOONCAKE_SOURCE = '''from dataclasses import dataclass

@dataclass(frozen=True)
class TransferRegion:
    layer_name: str
    layer_index: int
    base_addr: int
    block_len: int
    kv_block_len: int
    group_index: int = 0

def _get_tp_ratio(local_tp_size: int, remote_tp_size: int) -> int:
    if local_tp_size >= remote_tp_size:
        assert local_tp_size % remote_tp_size == 0
        return local_tp_size // remote_tp_size
    assert remote_tp_size % local_tp_size == 0
    return -(remote_tp_size // local_tp_size)

def _compute_sender_transfer_plan(
    local_tp_rank: int,
    local_tp_size: int,
    remote_tp_rank: int,
    remote_tp_size: int,
    local_kv_block_len: int,
    remote_kv_block_len: int,
    producer_cache_replicated: bool,
) -> tuple[bool, int, int, int]:
    tp_ratio = _get_tp_ratio(local_tp_size, remote_tp_size)
    if tp_ratio == 1:
        return True, 0, 0, local_kv_block_len
    if tp_ratio > 0:
        if producer_cache_replicated:
            return local_tp_rank % tp_ratio == 0, 0, 0, local_kv_block_len
        return True, 0, (local_tp_rank % tp_ratio) * local_kv_block_len, local_kv_block_len
    if producer_cache_replicated:
        return True, 0, 0, local_kv_block_len
    ratio_abs = -tp_ratio
    return True, (remote_tp_rank % ratio_abs) * remote_kv_block_len, 0, remote_kv_block_len

def _validate_asymmetric_region_lengths(
    local_regions: list[TransferRegion],
    remote_regions: list[TransferRegion],
    local_tp_size: int,
    remote_tp_size: int,
    producer_cache_replicated: bool,
) -> str | None:
    """Validate transfer-region metadata for a fixed producer/consumer pair.
    This checks registered KV regions, not per-request block counts. A region
    corresponds to one registered KV tensor, or one K/V half after expansion
    for layouts that store K and V together.
    """
    if len(local_regions) != len(remote_regions):
        return (
            "Mooncake asymmetric TP requires matching KV region counts between "
            "producer and consumer."
        )

    if producer_cache_replicated:
        return None
    tp_ratio = _get_tp_ratio(local_tp_size, remote_tp_size)
    for idx, (local_region, remote_region) in enumerate(
        zip(local_regions, remote_regions)
    ):
        if tp_ratio == 1:
            if local_region.kv_block_len != remote_region.kv_block_len:
                return (
                    "Mooncake KV region length mismatch for homogeneous TP at "
                    f"region {idx}: local={local_region.kv_block_len}, "
                    f"remote={remote_region.kv_block_len}."
                )
        elif tp_ratio > 0:
            if remote_region.kv_block_len != local_region.kv_block_len * tp_ratio:
                return (
                    "Mooncake destination KV region length does not match the "
                    "producer TP ratio at region "
                    f"{idx}: local={local_region.kv_block_len}, "
                    f"remote={remote_region.kv_block_len}, tp_ratio={tp_ratio}."
                )
        else:
            ratio_abs = -tp_ratio
            if local_region.kv_block_len != remote_region.kv_block_len * ratio_abs:
                return (
                    "Mooncake source KV region length does not match the "
                    "consumer TP ratio at region "
                    f"{idx}: local={local_region.kv_block_len}, "
                    f"remote={remote_region.kv_block_len}, tp_ratio={tp_ratio}."
                )
    return None

class MooncakeConnectorWorker:
    def _producer_cache_is_replicated(self) -> bool:
        return False

    def _get_sender_transfer_plan(
        self,
        local_kv_block_len: int,
        remote_kv_block_len: int,
        remote_tp_rank: int,
        remote_tp_size: int,
    ) -> tuple[bool, int, int, int]:
        return _compute_sender_transfer_plan(
            local_tp_rank=self.tp_rank,
            local_tp_size=self.tp_size,
            remote_tp_rank=remote_tp_rank,
            remote_tp_size=remote_tp_size,
            local_kv_block_len=local_kv_block_len,
            remote_kv_block_len=remote_kv_block_len,
            producer_cache_replicated=self._producer_cache_is_replicated(),
        )
'''


def _fake_vllm_tree(tmp_path: Path, mooncake_source: str) -> tuple[Path, Path]:
    vllm_root = tmp_path / "vllm"
    mooncake_target = vllm_root / "distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py"
    prom_target = vllm_root / "distributed/kv_transfer/kv_connector/v1/multi_connector.py"
    mooncake_target.parent.mkdir(parents=True)
    mooncake_target.write_text(mooncake_source)
    prom_target.write_text(OLD_BLOCK)
    (vllm_root / "__init__.py").write_text('__version__ = "test-version"\n')
    script = Path(__file__).parents[1] / "configs/patches/patch-vllm-mooncake-mixed-regions.sh"
    return mooncake_target, script


def _load_helpers(source: str) -> dict[str, object]:
    tree = ast.parse(source)
    wanted = {
        "TransferRegion",
        "_get_tp_ratio",
        "_compute_sender_transfer_plan",
        "_validate_asymmetric_region_lengths",
    }
    body = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            *body,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {"dataclass": dataclass}
    exec(compile(module, "<patched-mooncake>", "exec"), namespace)  # noqa: S102
    return namespace


def test_mooncake_mixed_region_patch_is_strict_idempotent_and_plans_per_region(
    tmp_path: Path,
):
    target, script = _fake_vllm_tree(tmp_path, OLD_MOONCAKE_SOURCE)
    env = os.environ | {
        "PYTHONPATH": str(tmp_path),
        "VLLM_MULTICONNECTOR_PROM_PATCH_SCRIPT": str(
            Path(__file__).parents[1] / "configs/patches/patch-vllm-multiconnector-prometheus-optional.sh"
        ),
    }

    first = subprocess.run(["bash", str(script)], env=env, text=True, capture_output=True, check=True)
    second = subprocess.run(["bash", str(script)], env=env, text=True, capture_output=True, check=True)
    patched = target.read_text()
    assert "VLLM_MOONCAKE_MIXED_REGION_PATCH=applied" in first.stdout
    assert "VLLM_MOONCAKE_MIXED_REGION_PATCH=already-applied" in second.stdout
    assert "local_kv_block_len == remote_kv_block_len" in patched

    helpers = _load_helpers(patched)
    region = helpers["TransferRegion"]
    validate = helpers["_validate_asymmetric_region_lengths"]
    plan = helpers["_compute_sender_transfer_plan"]
    local = [region("sharded", 0, 0, 0, 131072), region("replicated", 1, 0, 0, 65536)]
    remote = [region("sharded", 0, 0, 0, 65536), region("replicated", 1, 0, 0, 65536)]
    assert validate(local, remote, 2, 4, False) is None
    assert validate([region("bad", 0, 0, 0, 98304)], remote[:1], 2, 4, False)
    assert plan(0, 2, 1, 4, 65536, 65536, True) == (True, 0, 0, 65536)
    assert plan(0, 2, 1, 4, 131072, 65536, False) == (True, 65536, 0, 65536)


def test_mooncake_mixed_region_patch_rejects_unknown_source(tmp_path: Path):
    _, script = _fake_vllm_tree(tmp_path, "unexpected source\n")
    env = os.environ | {
        "PYTHONPATH": str(tmp_path),
        "VLLM_MULTICONNECTOR_PROM_PATCH_SCRIPT": str(
            Path(__file__).parents[1] / "configs/patches/patch-vllm-multiconnector-prometheus-optional.sh"
        ),
    }
    result = subprocess.run(
        ["bash", str(script)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Refusing to patch" in result.stderr
