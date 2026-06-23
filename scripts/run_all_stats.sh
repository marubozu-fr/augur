#!/usr/bin/env bash
#
# Run every standard stat family for a given instrument.
#
# Usage: ./scripts/run_all_stats.sh <INSTRUMENT>
# Example: ./scripts/run_all_stats.sh ES
#
# Discovers all stats/*/standard.py modules dynamically and runs each as
# `python -m stats.<family>.standard --instrument <INSTRUMENT>`.
# Exits non-zero if any module fails.

set -uo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <INSTRUMENT>" >&2
  exit 1
fi

instrument="$1"

# Resolve repository root from this script's location so it works from anywhere.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

config_file="config/${instrument}.yaml"
if [[ ! -f "${config_file}" ]]; then
  echo "Error: config not found: ${config_file}" >&2
  exit 1
fi

failed=()

for module_path in stats/*/standard.py; do
  [[ -e "${module_path}" ]] || continue
  family="$(basename "$(dirname "${module_path}")")"
  module="stats.${family}.standard"

  echo "==> ${module}"
  if ! python -m "${module}" --instrument "${instrument}"; then
    echo "FAILED: ${module}" >&2
    failed+=("${module}")
  fi
done

if [[ ${#failed[@]} -gt 0 ]]; then
  echo "" >&2
  echo "${#failed[@]} module(s) failed:" >&2
  for module in "${failed[@]}"; do
    echo "  - ${module}" >&2
  done
  exit 1
fi

echo "All stat families completed successfully."
