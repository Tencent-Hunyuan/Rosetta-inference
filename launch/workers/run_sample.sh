#!/usr/bin/env bash
# Runs a Rosetta torchrun evaluation/sampling job on a single node.
# Normally invoked by launch/run_multinode.sh, but can also be called directly.
#
# Usage (direct):
#   INDEX=0 HOST_NUM=1 HOST_GPU_NUM=8 FREE_PORT=23456 \
#       bash launch/workers/run_sample.sh torchrun <config> [extra args...]
#
# Arguments:
#   $1  Startup method: "torchrun" (default) or "python" (single-process debug)
#   $2  Path to YAML config file
#   $3+ Extra arguments forwarded to evaluation.entry

LAUNCH_DIR=$(realpath "$(dirname "$(dirname "$0")")")
source "${LAUNCH_DIR}/common_env.sh"
source "${LAUNCH_DIR}/common_tools.sh"

INDEX=${INDEX:-0}
HOST_NUM=${HOST_NUM:-1}
HOST_GPU_NUM=${HOST_GPU_NUM:-8}
_NAME="[NODE ${INDEX}]"

# ---------------------------------------------------------------------------
# Locate project root and set PYTHONPATH
# ---------------------------------------------------------------------------
PROJECT_BASE=$(dirname "${LAUNCH_DIR}")
print_msg "$_NAME PROJECT_BASE: ${PROJECT_BASE}"
cd "${PROJECT_BASE}" || exit 1

export PYTHONPATH="${PROJECT_BASE}:${PYTHONPATH}"
if [ -d "${PROJECT_BASE}/deps" ]; then
    while IFS= read -r -d '' subdir; do
        export PYTHONPATH="$(realpath "${subdir}"):${PYTHONPATH}"
        print_msg "$_NAME Added to PYTHONPATH: ${subdir}"
    done < <(find "${PROJECT_BASE}/deps" -mindepth 1 -maxdepth 1 -type d -print0)
fi

# ---------------------------------------------------------------------------
# Shared assets (VAE, ViT, tokenizer)
# Override: ASSETS_BASE=/path/to/your/assets bash scripts/eval/eval_ai2d.sh
# ---------------------------------------------------------------------------
export ASSETS_BASE="${ASSETS_BASE:-${PROJECT_BASE}/public_assets}"
print_msg "$_NAME ASSETS_BASE: ${ASSETS_BASE}"

# ---------------------------------------------------------------------------
# Parse startup method (torchrun / python)
# ---------------------------------------------------------------------------
STARTUP_METHOD="${1:-torchrun}"
if [ "${STARTUP_METHOD}" == "torchrun" ] || [ "${STARTUP_METHOD}" == "python" ]; then
    shift
else
    STARTUP_METHOD=torchrun
fi

# ---------------------------------------------------------------------------
# Parse config file
# ---------------------------------------------------------------------------
CONFIG_FILE="$1"
if [ -z "${CONFIG_FILE}" ]; then
    print_msg error "$_NAME Config file must be specified" >&2
    exit 1
fi
CONFIG_FILE=$(realpath "${CONFIG_FILE}")
if [ ! -f "${CONFIG_FILE}" ]; then
    print_msg error "$_NAME Config file not found: ${CONFIG_FILE}" >&2
    exit 1
fi
print_msg "$_NAME CONFIG: ${CONFIG_FILE}"
shift

# ---------------------------------------------------------------------------
# Output directory and logging
# ---------------------------------------------------------------------------
TASK_ID="${TASK_ID:-$(date +%Y%m%d-%H-%M-%S)}"
OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_BASE}}"
LOG_DIR="${OUTPUT_PATH}/outputs/logs/${TASK_ID}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${INDEX}.log"
print_msg "$_NAME OUTPUT_PATH: ${OUTPUT_PATH}"
print_msg "$_NAME LOG_FILE: ${LOG_FILE}"

# ---------------------------------------------------------------------------
# Master address (single-machine: always 127.0.0.1)
# ---------------------------------------------------------------------------
if [ "${INDEX}" -eq 0 ]; then
    MASTER_ADDR=127.0.0.1
else
    MASTER_ADDR="${CHIEF_IP:-127.0.0.1}"
fi
FREE_PORT="${FREE_PORT:-$(find_free_port)}"

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
case "${STARTUP_METHOD}" in
    torchrun)
        set -x
        torchrun \
            --nproc-per-node "${HOST_GPU_NUM}" \
            --nnodes "${HOST_NUM}" \
            --node-rank "${INDEX}" \
            --master-addr "${MASTER_ADDR}" \
            --master-port "${FREE_PORT}" \
            -m evaluation.entry \
            --config-path "${CONFIG_FILE}" \
            --sampler multimodal_sampler.MultimodalSampler \
            --task-id "${TASK_ID}" \
            "$@" 2>&1 | tee "${LOG_FILE}"
        ;;
    python)
        # Single-process mode for quick debugging (no distributed)
        set -x
        python3 -m evaluation.entry \
            --config-path "${CONFIG_FILE}" \
            --sampler multimodal_sampler.MultimodalSampler \
            --task-id "${TASK_ID}" \
            "$@" 2>&1 | tee "${LOG_FILE}"
        ;;
esac
