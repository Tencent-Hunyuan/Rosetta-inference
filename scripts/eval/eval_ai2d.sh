#!/usr/bin/env bash
# Evaluate AI2D (diagram understanding, 8 GPUs, instruct template)
# Usage:  bash scripts/eval/eval_ai2d.sh
# Override: CKPT_DIR=<hf_weights>  SAMPLE_OUT=<output>  bash scripts/eval/eval_ai2d.sh
# Need 9min

SCRIPT_DIR=$(realpath "$(dirname "$0")")
REPO_DIR=$(realpath "${SCRIPT_DIR}/../..")
LAUNCH_DIR="${REPO_DIR}/launch"
source "${LAUNCH_DIR}/common_env.sh"
source "${LAUNCH_DIR}/common_tools.sh"
_NAME="[eval_ai2d]"

export HOST_NUM=1
export HOST_GPU_NUM=${HOST_GPU_NUM:-8}
export OFFSET=${OFFSET:-0}
export GLOBAL_BATCH_SIZE=8

# Model selection (override EXP and CONFIG via env vars):
# # MoE-3.8B-A1B
# EXP="${EXP:-checkpoints/MoE-3.8B-A1B}"
# CONFIG="${CONFIG:-evaluation/configs/moe.yaml}"

# # MoT-4.5B-A1B
# EXP="${EXP:-checkpoints/MoT-4.5B-A1B}"
# CONFIG="${CONFIG:-evaluation/configs/mot.yaml}"

# Rosetta-3.8B-A1B (default)
EXP="${EXP:-checkpoints/Rosetta-3.8B-A1B}"
CONFIG="${CONFIG:-evaluation/configs/rosetta.yaml}"

CKPT_DIR="${CKPT_DIR:-${EXP}/hf_weights}"
SAMPLE_OUT="${SAMPLE_OUT:-${EXP}/outputs}"
export OUTPUT_PATH="${EXP}"

EVAL_METRICS="ai2d_test@@metric=ai2d@@max_new_tokens=5"
TARGET="${SAMPLE_OUT}/ai2d_test__ai2d_5/metric_results/ai2d.json"

ASSETS_BASE="${ASSETS_BASE:-${REPO_DIR}/public_assets}"

if [ ! -d "${CKPT_DIR}" ]; then
    print_msg "$_NAME Checkpoint not found: ${CKPT_DIR}"; exit 1
fi
if [ -f "${TARGET}" ]; then
    print_msg "$_NAME Already done: ${TARGET}"; exit 0
fi

print_msg "$_NAME Checkpoint : ${CKPT_DIR}"
print_msg "$_NAME Output     : ${SAMPLE_OUT}"

cd "${REPO_DIR}"
_t=$(date +%s)
bash launch/workers/run_sample.sh \
    torchrun \
    ${CONFIG} \
    --framework fsdp \
    --eval-metrics "${EVAL_METRICS}" \
    --sample-save-base "${SAMPLE_OUT}" \
    --ckpt "${CKPT_DIR}" \
    --sequence-template instruct \
    --generation-config "${ASSETS_BASE}/generation_configs/qwen3_0.6b_mmu_eval_instruct.json" \
    --sample-batch-size 1 \
    --no-do-sample \
    --reproduce \
    "$@"

print_msg "$_NAME Done in $(( ($(date +%s) - _t) / 60 ))m$(( ($(date +%s) - _t) % 60 ))s"
