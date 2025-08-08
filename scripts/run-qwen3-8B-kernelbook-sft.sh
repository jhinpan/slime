#!/bin/bash

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

export WANDB_API_KEY=0db9fd073cc9e49c8bcec2b0a6929792ecb64e4e

NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-8B.sh"

CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3-8B/
   --ref-load /root/Qwen3-8B_torch_dist
   --load /root/Qwen3-8B-KernelBook-SFT/
   --save /root/Qwen3-8B-KernelBook-SFT/
   --save-interval 500
)

SFT_ARGS=(
   --rollout-function-path slime.rollout.sft_rollout.generate_rollout
   --prompt-data /root/kernelbook_sft.jsonl
   --input-key messages
   --rollout-shuffle
   --num-epoch 1
   --rollout-batch-size 32
   --global-batch-size 32

   --loss-type sft_loss
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --debug-train-only
)

EVAL_ARGS=(
   --eval-interval 100
   --eval-prompt-data /root/kernelbook_eval.jsonl
   --eval-input-key messages
)

PERF_ARGS=(
   --tensor-model-parallel-size 8
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 2

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 5e-6
   --lr-warmup-iters 100
   --lr-decay-style cosine
   --min-lr 5e-7
   --lr-warmup-fraction 0.05
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.95
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-kernelbook
   --wandb-group qwen3-8B-kernelbook-sft
   --wandb-key ${WANDB_API_KEY}
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265


# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${SFT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${DISTRIBUTED_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${MISC_ARGS[@]}