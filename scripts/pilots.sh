#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
stamp="$(date +%Y%m%d-%H%M%S)"
for mode in L0 L1 L2 L3; do
  .venv/bin/ailife run --config configs/baselines/pilot.yaml --mode "$mode" --device "${AILIFE_DEVICE:-cpu}" --output "runs/${stamp}-${mode}"
done
