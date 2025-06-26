#!/bin/bash

# Kernel Code Generation Agent Training Script
# This script trains an agent to generate optimized CUDA kernels

# Clean up previous runs
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

export PYTHONBUFFERED=16

# Configure your WandB key if available
export WANDB_KEY=${WANDB_KEY:-"0db9fd073cc9e49c8bcec2b0a6929792ecb64e4e"}

# Model parallelism configuration
# Set TP_SIZE=1 to match the checkpoint configuration
export TP_SIZE=2
export PP_SIZE=1
export CP_SIZE=1

# Model paths
export HF_MODEL_PATH=/root/hf_models/deepseek-ai--DeepSeek-R1-Distill-Qwen-7B
export MCORE_MODEL_PATH=/root/megatron_model/DeepSeek-R1-Distill-Qwen-7B-25.02
export PROMPT_DATA=/root/slime/kb-rl-data/kb_hf_level_1_processed.jsonl
export MCORE_MODEL_PATH_SAVE=/root/megatron_model/DeepSeek-R1-Distill-Qwen-7B-kernel-agent

# DeepSeek-R1-Distill-Qwen-7B model architecture
MODEL_ARGS=(
   --swiglu
   --num-layers 28
   --hidden-size 3584
   --ffn-hidden-size 18944
   --num-attention-heads 28
   --group-query-attention
   --num-query-groups 4
   --max-position-embeddings 131072
   --seq-length 8192  # Increased for code generation
   --use-rotary-position-embeddings
   --disable-bias-linear
   --add-qkv-bias
   --normalization "RMSNorm"
   --norm-epsilon 1e-06
   --rotary-base 10000
   --vocab-size 152064
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --moe-token-dispatcher-type alltoall
   --untie-embeddings-and-output-weights
   --attention-dropout 0.0
   --hidden-dropout 0.0
)

CKPT_ARGS=(
   --hf-checkpoint ${HF_MODEL_PATH}
   --ref-load ${MCORE_MODEL_PATH}
   --save-interval 50  # Save more frequently for kernel training
   --save ${MCORE_MODEL_PATH_SAVE}
)

ROLLOUT_ARGS=(
   --rollout-function-path slime.rollout.agent_rollout.generate_rollout
   --rm-type deepscaler
   --prompt-data ${PROMPT_DATA}
   --label-key label
   --num-rollout 1000  # Reduced for kernel tasks
   --rollout-batch-size 128  # Adjusted for proper multiple
   --rollout-max-response-len 12288  # Larger for CUDA code
   --rollout-temperature 0.8  # Higher for code diversity
   --rollout-shuffle
   --n-samples-per-prompt 4  # Fewer samples per prompt for kernel code
   --global-batch-size 512  # Smaller global batch
   --micro-batch-size 4  # Smaller micro batch for longer sequences
   --ref-micro-batch-size 4
   --use-dynamic-batch-size
   --max-tokens-per-gpu 12288  # Increased for longer code sequences
   --balance-data
)

DISTRIBUTED_ARGS=(
   --tensor-model-parallel-size ${TP_SIZE}
   --pipeline-model-parallel-size ${PP_SIZE}
   --context-parallel-size ${CP_SIZE}
   --sequence-parallel
)

PERF_ARGS=(
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.005  # Slightly higher for code generation
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
)

OPTIMIZER_ARGS=(
   --lr 5e-7  # Lower learning rate for fine-tuning on code
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-kernel-agent
   --wandb-group DeepSeek-R1-Distill-Qwen-7B-KernelBench
   --wandb-key ${WANDB_KEY}
)

# Launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-"12345"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats

# Submit the training job
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "PYTHONPATH": "/root/Megatron-LM/",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_CUMEM_ENABLE": "0"
     }
   }' \
   -- python3 train_agent_async.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --rollout-num-gpus 4 \
   --rollout-num-gpus-per-engine 1 \
   --sglang-mem-fraction-static 0.7 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${DISTRIBUTED_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   --agent-rollout-buffer-url http://${MASTER_ADDR}:8889 \
   --keep-old-actor \
   --update-rollout-weights-interval 1 \
   --buffer-filter-path slime.rollout.filter_hub.buffer_filters.pop_first \
   --disable-rewards-normalization \
   --offload-rollout \
   --offload-ref \
   --rollout-input-file ${PROMPT_DATA} \
   --rollout-num-process 512 \
   --loss-mask-type distill_qwen \
   --sglang-log-level error \
   --input-key prompt \
   --log-passrate \
   --rollout-task-type kernelbench