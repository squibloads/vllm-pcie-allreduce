#!/usr/bin/env bash
# docker-run.sh - serve a model at TP=4 over PCIe P2P with 1Cat's push custom all-reduce for decode-sized
# all-reduces and NCCL for everything else. Pass verify/verify_car.sh on the same four GPUs first.
#
#   IMG=your-1cat-image GPUS=0,1,2,3 MODEL_DIR=/path/to/model PATCH=patched/custom_all_reduce.py \
#     examples/docker-run.sh [extra vllm serve args...]
#
# PATCH is car_patch.py's output for THIS image's custom_all_reduce.py (README, Install). A copy cut from a
# different 1Cat revision would silently revert that revision's other changes to the file.
set -euo pipefail
IMG=${IMG:?set IMG to your 1Cat-vLLM image}
GPUS=${GPUS:?set GPUS to the four GPUs: nvidia-smi indices or UUIDs, comma-separated}
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR to the model directory on the host}
PATCH=${PATCH:?set PATCH to car_patch.py output for this image}
NAME=${NAME:-vllm-tp4-pcie}
PORT=${PORT:-8000}
# a missing file must not reach docker, which would create a directory in its place
[ -f "$PATCH" ] || { echo "custom-AR patch $PATCH missing" >&2; exit 1; }
PATCH=$(realpath "$PATCH")
# where vllm lives inside the image, found without importing it
VLLM_PKG=${VLLM_PKG:-$(docker run --rm --entrypoint python3 "$IMG" -c \
  'import importlib.util, os; print(os.path.dirname(importlib.util.find_spec("vllm").origin))')}

CAR_ENV=(
  -e VLLM_CAR_P2P_MESH=1        # count the group as fully connected when every pair has P2P (all ranks agree)
  -e VLLM_CAR_MAX_BYTES=81921   # custom AR only below this: the push kernel covers up to 81920 bytes
  -e VLLM_CAR_GRAPH_ONLY=1      # custom AR only inside CUDA graphs: eager calls would take the pull kernel
  -e NCCL_P2P_LEVEL=SYS         # NCCL's own (large) all-reduces use P2P across root ports too
  -e VLLM_TP_ALLREDUCE_TRACE=1  # log which backend takes each all-reduce shape, once per shape
)
# 1Cat >= #821 (c4f6245f8 and later) routes greedy top-1 through a custom-AR pull kernel by default, outside
# what verify_car.sh tests (README, Known limits). TOP1_CUSTOM_AR=0 keeps it on the NCCL all-gather.
[ -n "${TOP1_CUSTOM_AR:-}" ] && CAR_ENV+=(-e "VLLM_SM70_TOP1_CUSTOM_AR=$TOP1_CUSTOM_AR")

# The model flags we ran Qwen3.8-Flash-Next NVFP4 with on 1Cat 357d07bcb / c4f6245f8. They are not needed
# for the all-reduce; take yours from 1Cat's documentation for your model and revision.
MODEL_ENV=(
  -e VLLM_SM70_FLASH_ATTN_V100=1 -e VLLM_SM70_QWEN38_FP16_GEMV=1
  -e VLLM_SM70_QWEN38_FUSED_HC_FP16=1 -e VLLM_SM70_QWEN38_FUSED_GDN_INPUT_FP16=1
  -e VLLM_QWEN4EXP_PLE_HOST_GIB=12
)
SERVE_ARGS=(
  --max-model-len 262144 --max-num-seqs 2 --max-num-batched-tokens 2048 --gpu-memory-utilization 0.90
  --enable-prefix-caching --mamba-cache-mode align
)

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --ipc=host --ulimit memlock=-1 --gpus "\"device=$GPUS\"" \
  "${CAR_ENV[@]}" "${MODEL_ENV[@]}" \
  -v "$PATCH:$VLLM_PKG/distributed/device_communicators/custom_all_reduce.py:ro" \
  -v "$MODEL_DIR:/model:ro" -p "127.0.0.1:$PORT:8000" \
  "$IMG" vllm serve /model --tensor-parallel-size 4 "${SERVE_ARGS[@]}" "$@" \
  --host 0.0.0.0 --port 8000 >/dev/null
echo "$NAME started; patch $PATCH mounted over $VLLM_PKG/distributed/device_communicators/custom_all_reduce.py"
echo "check: docker logs $NAME 2>&1 | grep -E 'VLLM_CAR_P2P_MESH|push all-reduce enabled|all-reduce backends'"
