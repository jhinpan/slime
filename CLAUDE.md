# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

**slime** is an LLM post-training framework focusing on RL scaling. It connects Megatron with SGLang for high-performance training and provides flexible data generation interfaces for arbitrary training workflows.

## Commands

### Installation
```bash
pip install -e .
```

### Code Quality
```bash
# Run pre-commit hooks (autoflake, isort, black)
pre-commit run --all-files

# Run specific formatters
black . --line-length 119
isort . --profile black --line-length 119
```

### Testing
```bash
# Run tests with pytest
pytest tests/

# Run tests with specific markers
pytest -m "unit"
pytest -m "not integration"
```

### Training
```bash
# Start Ray cluster (on head node)
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats

# Submit training job
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "PYTHONPATH": "/root/Megatron-LM/"
     }
   }' \
   -- python3 train.py \
   --tensor-model-parallel-size 2 \
   --hf-checkpoint /path/to/model \
   # ... other arguments
```

### Checkpoint Conversion
```bash
# HF to Megatron torch_dist
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    --hf-checkpoint /path/to/hf_model \
    --save /path/to/torch_dist_model

# Megatron torch_dist to HF
PYTHONPATH=/root/Megatron-LM python tools/convert_torch_dist_to_hf.py \
  --input-dir /path/to/torch_dist_ckpt/iter_xxx/ \
  --output-dir /path/to/hf_output \
  --origin-hf-dir /path/to/original_hf_model
```

## Architecture

The system consists of three main components:

1. **Training Actor (Megatron)**: Handles the main training loop, reads from data buffer, manages model parallelism
   - Located in `slime/backends/megatron_utils/`
   - Key files: `model.py`, `loss.py`, `data.py`, `checkpoint.py`

2. **Rollout Generator (SGLang + router)**: Generates new training data with rewards/verifier outputs
   - Located in `slime/backends/sglang_utils/` and `slime/ray/rollout.py`
   - Manages multiple SGLang engines for inference
   - Updates weights from training actor before generation

3. **Data Buffer**: Bridge between training and rollout generation
   - Located in `slime/ray/buffer.py`
   - Manages prompt initialization and custom data generation
   - Supports save/load for global dataset persistence

### Key Design Patterns

- **Ray-based Distribution**: All components run as Ray actors for distributed execution
- **Placement Groups**: Manages GPU allocation between training and inference
- **Async Operations**: Training and rollout can be made asynchronous by adjusting `ray.get()` positions
- **Weight Synchronization**: Automatic parameter updates from Megatron to SGLang before each rollout

### Arguments Structure

1. **Megatron arguments**: Standard Megatron-LM arguments (e.g., `--tensor-model-parallel-size`)
2. **SGLang arguments**: Prefixed with `--sglang-` (e.g., `--sglang-mem-fraction-static`)
3. **slime arguments**: Framework-specific arguments defined in `slime/utils/arguments.py`

## Important Considerations

- Always set `PYTHONPATH` to Megatron-LM directory when running commands
- The framework requires both Megatron and SGLang to be properly installed
- Use the provided Docker image `zhuzilin/slime:latest` for the easiest setup
- When using `--colocate`, training and inference share GPUs, automatically enabling `--offload`