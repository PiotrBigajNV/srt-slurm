# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
import subprocess


OLD_BLOCK = '''    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        for connector_id, stats_data in transfer_stats_data.items():
            assert connector_id in self._prom_metrics, (
                f"{connector_id} is not contained in the list of registered connectors "
                f"with Prometheus metrics support: {self._prom_metrics.keys()}"
            )
            self._prom_metrics[connector_id].observe(stats_data["data"], engine_idx)
'''


def _fake_vllm_tree(tmp_path: Path, source: str) -> tuple[Path, Path]:
    vllm_root = tmp_path / "vllm"
    target = (
        vllm_root
        / "distributed/kv_transfer/kv_connector/v1/multi_connector.py"
    )
    target.parent.mkdir(parents=True)
    target.write_text(source)
    (vllm_root / "__init__.py").write_text('__version__ = "test-version"\n')
    script = (
        Path(__file__).parents[1]
        / "configs/patches/patch-vllm-multiconnector-prometheus-optional.sh"
    )
    return target, script


def test_multiconnector_prometheus_patch_is_strict_and_idempotent(tmp_path: Path):
    target, script = _fake_vllm_tree(tmp_path, OLD_BLOCK)
    env = os.environ | {"PYTHONPATH": str(tmp_path)}

    first = subprocess.run(
        ["bash", str(script)], env=env, text=True, capture_output=True, check=True
    )
    second = subprocess.run(
        ["bash", str(script)], env=env, text=True, capture_output=True, check=True
    )

    patched = target.read_text()
    assert "connector_prom = self._prom_metrics.get(connector_id)" in patched
    assert "if connector_prom is None:" in patched
    assert "assert connector_id in self._prom_metrics" not in patched
    assert "VLLM_MULTICONNECTOR_PROM_PATCH=applied" in first.stdout
    assert "VLLM_MULTICONNECTOR_PROM_PATCH=already-applied" in second.stdout


def test_multiconnector_prometheus_patch_rejects_unknown_source(tmp_path: Path):
    _, script = _fake_vllm_tree(tmp_path, "unexpected source\n")
    env = os.environ | {"PYTHONPATH": str(tmp_path)}

    result = subprocess.run(
        ["bash", str(script)], env=env, text=True, capture_output=True
    )

    assert result.returncode != 0
    assert "Refusing to patch" in result.stderr
