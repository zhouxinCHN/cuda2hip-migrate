#!/usr/bin/env bash
# Run the strict target-environment check from the project root.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
exec python3 -m rocm10_samples_compat check-env --strict "$@"
