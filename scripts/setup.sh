#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
if ! command -v uv >/dev/null; then
  printf '%s\n' 'uv is required. Install it from https://docs.astral.sh/uv/getting-started/installation/' >&2
  exit 1
fi
if [[ ! -x .venv/bin/python ]]; then uv venv .venv --python python3.12; fi
if [[ "${1:-rocm}" == cpu ]]; then
  uv pip install --python .venv/bin/python torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
  uv pip install --python .venv/bin/python -e '.[test]'
else
  uv pip install --python .venv/bin/python -r requirements-rocm.lock --index-url https://repo.amd.com/rocm/whl-multi-arch/ --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
  uv pip install --python .venv/bin/python -e '.[test]'
fi
printf '%s\n' 'Environment ready. Browser tests additionally need: .venv/bin/playwright install chromium'
