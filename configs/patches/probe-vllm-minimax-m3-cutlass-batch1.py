# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Probe MiniMax-M3 CUTLASS sparse decode at the FULL-graph size-4 shape.

This intentionally calls the CUTLASS implementation directly, bypassing its
batch-size dispatch threshold.  It checks eager execution and CUDA graph
capture/replay against a small PyTorch reference without invoking the Triton
sparse fallback that this diagnostic is intended to avoid.
"""

from __future__ import annotations

import math

import torch
from vllm.models.minimax_m3.nvidia.msa_cutlass_sparse_decode import (
    MSACutlassDecodePlanCache,
    msa_cutlass_sparse_decode,
    prepare_decode_metadata,
)

HEAD_DIM = 128
PAGE_SIZE = 128
TOPK = 16
QUERY_LEN = 4
NUM_Q_HEADS = 16
NUM_KV_HEADS = 1
SM_SCALE = HEAD_DIM**-0.5


def _make_topk(seq_len: int, device: torch.device) -> torch.Tensor:
    topk = torch.full(
        (QUERY_LEN, NUM_KV_HEADS, TOPK),
        -1,
        dtype=torch.int32,
        device=device,
    )
    for local_query in range(QUERY_LEN):
        visible_tokens = seq_len - QUERY_LEN + local_query + 1
        visible_pages = math.ceil(visible_tokens / PAGE_SIZE)
        topk[local_query, :, :visible_pages] = torch.arange(
            visible_pages,
            dtype=torch.int32,
            device=device,
        )
    return topk


def _reference(
    query_fp8: torch.Tensor,
    kv_cache: torch.Tensor,
    topk: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    """Small GQA reference for one request with one KV head."""
    output = torch.empty(
        QUERY_LEN,
        NUM_Q_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=query_fp8.device,
    )
    key = kv_cache[..., :HEAD_DIM]
    value = kv_cache[..., HEAD_DIM:]

    for local_query in range(QUERY_LEN):
        query_position = seq_len - QUERY_LEN + local_query
        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        for logical_page_tensor in topk[local_query, 0]:
            logical_page = int(logical_page_tensor.item())
            if logical_page < 0:
                continue
            physical_page = int(block_table[0, logical_page].item())
            valid_tokens = min(
                PAGE_SIZE,
                query_position - logical_page * PAGE_SIZE + 1,
            )
            if valid_tokens <= 0:
                continue
            keys.append(key[physical_page, 0, :valid_tokens].float())
            values.append(value[physical_page, 0, :valid_tokens].float())

        gathered_key = torch.cat(keys, dim=0)
        gathered_value = torch.cat(values, dim=0)
        scores = torch.matmul(query_fp8[local_query].float(), gathered_key.T)
        probabilities = torch.softmax(scores * SM_SCALE, dim=-1)
        output[local_query].copy_(
            torch.matmul(probabilities, gathered_value).to(torch.bfloat16)
        )
    return output


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, phase: str) -> None:
    actual_float = actual.float()
    expected_float = expected.float()
    max_abs = float((actual_float - expected_float).abs().max().item())
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
    print(
        "VLLM_MINIMAX_M3_CUTLASS_BATCH1_PROBE_DETAIL "
        f"phase={phase} max_abs={max_abs:.8f}",
        flush=True,
    )


def main() -> None:
    if not torch.cuda.is_available():
        print("VLLM_MINIMAX_M3_CUTLASS_BATCH1_PROBE=skipped-no-cuda", flush=True)
        return

    torch.manual_seed(0)
    device = torch.device("cuda")
    capability = torch.cuda.get_device_capability(device)
    if capability[0] != 10:
        raise RuntimeError(
            "MiniMax-M3 CUTLASS batch-1 probe requires an SM100-family GPU; "
            f"got capability={capability}"
        )

    # One batch with query length four is the exact FULL descriptor that
    # selects Triton under the stock minimum-batch threshold.
    seq_len_initial = 257
    seq_lens_cpu = torch.tensor([seq_len_initial], dtype=torch.int32)
    seq_lens = seq_lens_cpu.to(device)
    block_table = torch.tensor([[2, 0, 1]], dtype=torch.int32, device=device)

    key = (
        torch.randn(
            3,
            NUM_KV_HEADS,
            PAGE_SIZE,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.25
    ).to(torch.float8_e4m3fn)
    value = (
        torch.randn_like(key, dtype=torch.bfloat16) * 0.25
    ).to(torch.float8_e4m3fn)
    kv_cache = torch.cat((key, value), dim=-1)
    query_fp8 = torch.randn(
        QUERY_LEN,
        NUM_Q_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    ).to(torch.float8_e4m3fn)
    topk = _make_topk(seq_len_initial, device)

    plan_cache = MSACutlassDecodePlanCache()
    metadata = prepare_decode_metadata(
        block_table,
        seq_lens,
        seq_lens_cpu,
        QUERY_LEN,
        num_q_heads=NUM_Q_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        page_size=PAGE_SIZE,
        topk_blocks=TOPK,
        plan_cache=plan_cache,
    )
    output = torch.empty(
        QUERY_LEN,
        NUM_Q_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    msa_cutlass_sparse_decode(
        query_fp8,
        kv_cache,
        topk,
        output,
        metadata,
        scale=SM_SCALE,
        q_scale_float=1.0,
        k_scale_float=1.0,
        v_scale_float=1.0,
    )
    torch.cuda.synchronize(device)
    _assert_close(
        output,
        _reference(query_fp8, kv_cache, topk, block_table, seq_len_initial),
        "eager",
    )

    # Capture the same batch-1/query-length-4 CUTLASS plan, then mutate its
    # graph-stable runtime metadata and replay it at a different ragged length.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        msa_cutlass_sparse_decode(
            query_fp8,
            kv_cache,
            topk,
            output,
            metadata,
            scale=SM_SCALE,
            q_scale_float=1.0,
            k_scale_float=1.0,
            v_scale_float=1.0,
        )

    replay_seq_len = 129
    replay_seq_lens_cpu = torch.tensor([replay_seq_len], dtype=torch.int32)
    seq_lens.copy_(replay_seq_lens_cpu.to(device))
    topk.copy_(_make_topk(replay_seq_len, device))
    replay_metadata = prepare_decode_metadata(
        block_table,
        seq_lens,
        replay_seq_lens_cpu,
        QUERY_LEN,
        num_q_heads=NUM_Q_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        page_size=PAGE_SIZE,
        topk_blocks=TOPK,
        plan_cache=plan_cache,
    )
    if replay_metadata.plan is not metadata.plan:
        raise RuntimeError("CUTLASS batch-1 plan cache was not graph-stable")
    graph.replay()
    torch.cuda.synchronize(device)
    _assert_close(
        output,
        _reference(query_fp8, kv_cache, topk, block_table, replay_seq_len),
        "graph-replay",
    )
    print(
        "VLLM_MINIMAX_M3_CUTLASS_BATCH1_PROBE=passed "
        "batch=1 query_len=4 tp=4 kv_dtype=fp8_e4m3 page_size=128 topk=16",
        flush=True,
    )


if __name__ == "__main__":
    main()
