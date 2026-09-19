#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="${SUPERTONIC_APP_ROOT:-${HOME}/supertonic-tts}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SUPERTONIC_REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
MODEL_ROOT="${SUPERTONIC_MODEL_ROOT:-${APP_ROOT}/models}"
OUTPUT_ROOT="${SUPERTONIC_OUTPUT_ROOT:-${APP_ROOT}/output}"
CACHE_ROOT="${SUPERTONIC_CACHE_ROOT:-${XDG_CACHE_HOME:-${HOME}/.cache}/supertonic-tts}"
VENV="${APP_ROOT}/.venv"

export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${CACHE_ROOT}/pip}"
export TMPDIR="${TMPDIR:-${CACHE_ROOT}/tmp}"

mkdir -p \
  "${APP_ROOT}" \
  "${OUTPUT_ROOT}" \
  "${MODEL_ROOT}" \
  "${HF_HOME}" \
  "${PIP_CACHE_DIR}" \
  "${TMPDIR}"

if ! command -v sima-cli >/dev/null 2>&1; then
  echo "sima-cli is required to download PyNeat" >&2
  exit 2
fi
if [[ ! -f "${REPO_ROOT}/requirements-devkit.txt" ]]; then
  echo "Missing requirements file under ${REPO_ROOT}" >&2
  exit 2
fi

python3 -m venv --clear "${VENV}"

PYNEAT_DOWNLOAD_DIR="$(mktemp -d "${TMPDIR}/pyneat-download.XXXXXX")"
cleanup() {
  rm -rf "${PYNEAT_DOWNLOAD_DIR}"
}
trap cleanup EXIT

sima-cli neat install core -t pyneat --install-dir "${PYNEAT_DOWNLOAD_DIR}"

mapfile -t PYNEAT_WHEELS < <(
  find "${PYNEAT_DOWNLOAD_DIR}" -maxdepth 1 -type f -name 'pyneat-*.whl' -print \
    | sort
)
if [[ "${#PYNEAT_WHEELS[@]}" -ne 1 ]]; then
  echo \
    "Expected one PyNeat wheel from sima-cli, found ${#PYNEAT_WHEELS[@]}" \
    >&2
  exit 2
fi

"${VENV}/bin/python" -m pip install --upgrade pip
"${VENV}/bin/python" -m pip install -r "${REPO_ROOT}/requirements-devkit.txt"
"${VENV}/bin/python" -m pip install --no-deps "${PYNEAT_WHEELS[0]}"

SUPERTONIC_APP_ROOT="${APP_ROOT}" \
  SUPERTONIC_MODEL_ROOT="${MODEL_ROOT}" \
  SUPERTONIC_REPO_ROOT="${REPO_ROOT}" \
  bash "${REPO_ROOT}/scripts/download_models.sh"

"${VENV}/bin/python" - <<'PY'
import numpy
import onnxruntime
import pyneat

print(f"numpy={numpy.__version__}")
print(f"onnxruntime={onnxruntime.__version__}")
print(f"pyneat={getattr(pyneat, '__version__', 'unknown')}")
PY

echo "DevKit environment ready: ${VENV}"
echo "Model assets ready: ${MODEL_ROOT}"
echo "Output directory ready: ${OUTPUT_ROOT}"
