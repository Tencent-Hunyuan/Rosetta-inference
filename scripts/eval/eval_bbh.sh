#!/usr/bin/env bash
# Evaluate BBH (generation, 8 GPUs, max_new_tokens=128)
# Usage:  bash scripts/eval/eval_bbh.sh
# Override: CKPT_DIR=<hf_weights>  SAMPLE_OUT=<output>  bash scripts/eval/eval_bbh.sh
# Need 170min

SCRIPT_DIR=$(realpath "$(dirname "$0")")
REPO_DIR=$(realpath "${SCRIPT_DIR}/../..")
LAUNCH_DIR="${REPO_DIR}/launch"
source "${LAUNCH_DIR}/common_env.sh"
source "${LAUNCH_DIR}/common_tools.sh"
_NAME="[eval_bbh]"

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

EVAL_METRICS="bbh@@metric=bbh@@max_new_tokens=128"
TARGET="${SAMPLE_OUT}/bbh__bbh_128/metric_results/bbh.json"

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
    --sequence-template pretrain \
    --generation-config "${ASSETS_BASE}/generation_configs/qwen3_0.6b_lm_eval.json" \
    --sample-batch-size 1 \
    --no-do-sample \
    --bot-task auto \
    --reproduce \
    "$@"

print_msg "$_NAME Done in $(( ($(date +%s) - _t) / 60 ))m$(( ($(date +%s) - _t) % 60 ))s"
