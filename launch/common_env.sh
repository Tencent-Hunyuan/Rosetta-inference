# Generic environment settings for Rosetta training and evaluation.
# These settings work on any standard multi-GPU machine.

# ---------------------------------------------------------------------------
# PyTorch / CUDA
# ---------------------------------------------------------------------------
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1

# ---------------------------------------------------------------------------
# NCCL
# ---------------------------------------------------------------------------
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export TORCH_NCCL_ENABLE_MONITORING=0

# ---------------------------------------------------------------------------
# HuggingFace / Loguru
# ---------------------------------------------------------------------------
export TOKENIZERS_PARALLELISM=false
export LOGURU_COLORIZE=true

# ---------------------------------------------------------------------------
# Utility: find an available TCP port for torchrun master
# ---------------------------------------------------------------------------
find_free_port() {
    local start=23456
    local end=33456
    local port
    for port in $(seq $start 100 $end); do
        (echo >/dev/tcp/localhost/$port) >/dev/null 2>&1
        if [[ $? -eq 1 ]]; then
            echo $port
            return
        fi
    done
    echo $start
}
