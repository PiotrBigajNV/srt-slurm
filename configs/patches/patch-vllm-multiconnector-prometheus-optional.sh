#!/usr/bin/env bash
set -euo pipefail

VLLM_ROOT="$(python3 - <<'PY'
from pathlib import Path

import vllm

print(Path(vllm.__file__).resolve().parent)
PY
)"
TARGET="${VLLM_ROOT}/distributed/kv_transfer/kv_connector/v1/multi_connector.py"

python3 - "${TARGET}" <<'PY'
from pathlib import Path
import sys

target = Path(sys.argv[1])
source = target.read_text()

old = '''    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        for connector_id, stats_data in transfer_stats_data.items():
            assert connector_id in self._prom_metrics, (
                f"{connector_id} is not contained in the list of registered connectors "
                f"with Prometheus metrics support: {self._prom_metrics.keys()}"
            )
            self._prom_metrics[connector_id].observe(stats_data["data"], engine_idx)
'''
new = '''    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        for connector_id, stats_data in transfer_stats_data.items():
            connector_prom = self._prom_metrics.get(connector_id)
            if connector_prom is None:
                continue
            connector_prom.observe(stats_data["data"], engine_idx)
'''

if old in source:
    target.write_text(source.replace(old, new, 1))
    state = "applied"
elif new in source:
    state = "already-applied"
else:
    raise SystemExit(
        f"Refusing to patch {target}: expected MultiConnector Prometheus "
        "observer block was not found"
    )

verified = target.read_text()
if new not in verified or old in verified:
    raise SystemExit(f"MultiConnector Prometheus patch verification failed for {target}")

print(f"VLLM_MULTICONNECTOR_PROM_PATCH={state}")
print(f"VLLM_MULTICONNECTOR_PROM_PATCH_TARGET={target}")
PY

python3 - <<'PY'
import vllm

print(f"VLLM_MULTICONNECTOR_PROM_PATCH_VLLM_VERSION={vllm.__version__}")
PY
