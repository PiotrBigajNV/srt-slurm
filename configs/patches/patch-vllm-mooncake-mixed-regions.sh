#!/usr/bin/env bash
set -euo pipefail

PROM_PATCH="${VLLM_MULTICONNECTOR_PROM_PATCH_SCRIPT:-/configs/patches/patch-vllm-multiconnector-prometheus-optional.sh}"
bash "${PROM_PATCH}"

VLLM_ROOT="$(python3 - <<'PY'
from pathlib import Path

import vllm

print(Path(vllm.__file__).resolve().parent)
PY
)"
TARGET="${VLLM_ROOT}/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py"

python3 - "${TARGET}" <<'PY'
from pathlib import Path
import sys

target = Path(sys.argv[1])
source = target.read_text()

old_validation = '''def _validate_asymmetric_region_lengths(
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
'''
new_validation = '''def _validate_asymmetric_region_lengths(
    local_regions: list[TransferRegion],
    remote_regions: list[TransferRegion],
    local_tp_size: int,
    remote_tp_size: int,
    producer_cache_replicated: bool,
) -> str | None:
    """Validate mixed replicated and TP-sharded KV transfer regions.

    MiniMax can register replicated regions alongside TP-sharded attention
    regions. Infer each region's transfer mode from its byte geometry instead
    of applying the model-wide replication flag to every registered region.
    The argument remains in the signature for compatibility with the pinned
    vLLM caller.
    """
    del producer_cache_replicated
    if len(local_regions) != len(remote_regions):
        return (
            "Mooncake asymmetric TP requires matching KV region counts between "
            "producer and consumer."
        )

    tp_ratio = _get_tp_ratio(local_tp_size, remote_tp_size)
    for idx, (local_region, remote_region) in enumerate(
        zip(local_regions, remote_regions)
    ):
        local_len = local_region.kv_block_len
        remote_len = remote_region.kv_block_len
        if local_len == remote_len:
            # This region is replicated on both sides.
            continue
        if tp_ratio > 0 and remote_len == local_len * tp_ratio:
            # The producer region is a TP shard of the consumer region.
            continue
        if tp_ratio < 0 and local_len == remote_len * -tp_ratio:
            # The consumer region is a TP shard of the producer region.
            continue
        return (
            "Mooncake KV region is neither replicated nor TP-sharded at region "
            f"{idx}: local={local_len}, remote={remote_len}, "
            f"tp_ratio={tp_ratio}."
        )
    return None
'''

old_plan = '''    def _get_sender_transfer_plan(
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
new_plan = '''    def _get_sender_transfer_plan(
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
            # MiniMax mixes replicated and TP-sharded KV regions. Equal byte
            # lengths identify a replicated region; scaled lengths identify a
            # TP shard and are validated before this method is called.
            producer_cache_replicated=(
                local_kv_block_len == remote_kv_block_len
            ),
        )
'''

validation_old_present = old_validation in source
validation_new_present = new_validation in source
plan_old_count = source.count(old_plan)
plan_new_present = new_plan in source

if validation_old_present and plan_old_count == 1:
    source = source.replace(old_validation, new_validation, 1)
    source = source.replace(old_plan, new_plan, 1)
    target.write_text(source)
    state = "applied"
elif validation_new_present and plan_new_present and plan_old_count == 0:
    state = "already-applied"
else:
    raise SystemExit(
        f"Refusing to patch {target}: expected Mooncake mixed-region "
        "validation/planner blocks were not found in a known state"
    )

verified = target.read_text()
if (
    new_validation not in verified
    or new_plan not in verified
    or old_validation in verified
    or old_plan in verified
):
    raise SystemExit(f"Mooncake mixed-region patch verification failed for {target}")

compile(verified, str(target), "exec")
print(f"VLLM_MOONCAKE_MIXED_REGION_PATCH={state}")
print(f"VLLM_MOONCAKE_MIXED_REGION_PATCH_TARGET={target}")
PY

python3 - <<'PY'
import vllm

print(f"VLLM_MOONCAKE_MIXED_REGION_PATCH_VLLM_VERSION={vllm.__version__}")
PY
