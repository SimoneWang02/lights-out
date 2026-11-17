#!/usr/bin/env bash
# Run the whole pipeline locally, in order. Usage: ./run_all.sh [first_stage] [last_stage]
#   ./run_all.sh        all stages (1-8)
#   ./run_all.sh 5      from the effect estimates onward
#   ./run_all.sh 2 4    cleaning through the exposure table
set -euo pipefail
cd "$(dirname "$0")"
first="${1:-1}"
last="${2:-8}"
for job in jobs/0[1-8]_*.py; do
  n="$(basename "$job" | cut -c2)"
  if [ "$n" -ge "$first" ] && [ "$n" -le "$last" ]; then
    echo "=== $job ==="
    uv run python "$job" --conf "${LIGHTSOUT_CONF:-conf/local.yaml}"
  fi
done
