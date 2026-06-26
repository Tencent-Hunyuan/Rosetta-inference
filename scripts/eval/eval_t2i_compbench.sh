#!/usr/bin/env bash
# Evaluate text-to-image generation (T2I-CompBench, 8 GPUs, batch=32)
# Usage:  bash scripts/eval/eval_t2i_compbench.sh
# Override: CKPT_DIR=<hf_weights>  SAMPLE_OUT=<output>  bash scripts/eval/eval_t2i_compbench.sh
# Need 12min

SCRIPT_DIR=$(realpath "$(dirname "$0")")
REPO_DIR=$(realpath "${SCRIPT_DIR}/../..")
LAUNCH_DIR="${REPO_DIR}/launch"
source "${LAUNCH_DIR}/common_env.sh"
source "${LAUNCH_DIR}/common_tools.sh"
_NAME="[eval_t2i]"

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

TESTSETS="t2i_compbench"
T2I_IMAGE_DIR="${SAMPLE_OUT}/t2i_compbench"
SCORE_RESULT="${T2I_IMAGE_DIR}/metric_results/result.json"

ASSETS_BASE="${ASSETS_BASE:-${REPO_DIR}/public_assets}"

if [ ! -d "${CKPT_DIR}" ]; then
    print_msg "$_NAME Checkpoint not found: ${CKPT_DIR}"; exit 1
fi
if [ -f "${SCORE_RESULT}" ]; then
    print_msg "$_NAME Already done: ${SCORE_RESULT}"; exit 0
fi

print_msg "$_NAME Checkpoint : ${CKPT_DIR}"
print_msg "$_NAME Output     : ${SAMPLE_OUT}"

cd "${REPO_DIR}"
_t=$(date +%s)

# Step 1: generate images
bash launch/workers/run_sample.sh \
    torchrun \
    ${CONFIG} \
    --framework fsdp \
    --testsets "${TESTSETS}" \
    --sample-save-base "${SAMPLE_OUT}" \
    --ckpt "${CKPT_DIR}" \
    --generation-config "${ASSETS_BASE}/generation_configs/qwen3_0.6b_t2i_eval.json" \
    --sequence-template instruct \
    --sample-batch-size 32 \
    --image-size 1:1 \
    --reproduce \
    --eval-save-images \
    "$@"

# Step 2: offline scoring (BLIP-VQA, single GPU)
print_msg "$_NAME Running offline T2I-CompBench scoring..."
PYTHONPATH="${REPO_DIR}" python3 -m evaluation.metrics.t2i_compbench.offline \
    --image-dir "${T2I_IMAGE_DIR}" \
    --batch-size 64

print_msg "$_NAME Done in $(( ($(date +%s) - _t) / 60 ))m$(( ($(date +%s) - _t) % 60 ))s"
