#!/usr/bin/env bash
# Usage: source /media/camp/EXT_DRIVE/rm75_control/env.sh

RM75_ENV="/media/camp/EXT_DRIVE/envs/rm75"

if [ ! -d "${RM75_ENV}/bin" ]; then
  echo "rm75 env not found: ${RM75_ENV}" >&2
  return 1 2>/dev/null || exit 1
fi

# Prefer direct PATH (works even if conda name lookup fails)
export PATH="${RM75_ENV}/bin:${PATH}"

# Optional: also hook conda if available
CONDA_BASE="${CONDA_BASE:-/home/camp/miniconda3}"
if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${RM75_ENV}" 2>/dev/null || true
fi

export RM75_CONTROL_ROOT="/media/camp/EXT_DRIVE/rm75_control"
export RM_API2_PYTHON="/media/camp/EXT_DRIVE/RM_API2/Python"
export PYTHONPATH="${RM75_CONTROL_ROOT}:${RM_API2_PYTHON}:${PYTHONPATH:-}"

echo "rm75 env: $(which python)"
echo "PYTHONPATH includes RM_API2 and rm75_control"
