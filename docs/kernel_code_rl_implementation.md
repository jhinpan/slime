# Kernel Code RL Implementation Guide

## Overview

This document provides a comprehensive guide for the Kernel Code Reinforcement Learning implementation in the Slime framework. The system trains an agent to generate optimized CUDA kernels for PyTorch operations using reinforcement learning techniques.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Slime Training                        │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Actor     │────│  SGLang      │────│   Rollout    │  │
│  │   Model     │    │  Server      │    │   Buffer     │  │
│  └─────────────┘    └──────────────┘    └──────────────┘  │
│         ↑                                       │           │
│         │                                       ↓           │
│  ┌─────────────┐                        ┌──────────────┐  │
│  │  Training   │←───────────────────────│   Kernel     │  │
│  │   Loop      │                        │  Generator   │  │
│  └─────────────┘                        └──────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. Data Preparation

#### Data Conversion Tool
- **Location**: `/root/slime/slime_plugins/rollout_buffer/tools/convert_kb_to_slime_format.py`
- **Purpose**: Converts KernelBench format data to Slime's expected format
- **Usage**:
  ```bash
  python ./slime_plugins/rollout_buffer/tools/convert_kb_to_slime_format.py \
      --input_path /root/kb-rl-data/kb_hf_level_1.jsonl \
      --task_type kernelbench \
      --output_path /root/kb-rl-data/kb_hf_level_1_processed.jsonl
  ```

#### Data Format
- **Input Format** (KernelBench):
  ```json
  {
      "instruction": "You are an expert in writing CUDA Kernels...",
      "input": "You write custom CUDA kernels to replace...",
      "output": "",
      "problem_id": 100
  }
  ```

- **Output Format** (Slime):
  ```json
  {
      "prompt": [
          {"role": "system", "content": "You are an expert..."},
          {"role": "user", "content": "You write custom..."}
      ],
      "label": "",
      "instance_id": "kernelbench_100"
  }
  ```

### 2. Kernel Generator

#### File Structure
```
slime_plugins/rollout_buffer/generator/
├── kernel_generator.py          # Main generator implementation
├── reward_utils/
│   └── kernel_utils.py         # Reward calculation functions
└── utils/
    └── default_func.py         # Default processing functions
```

#### Key Features
- **Asynchronous rollout generation** for CUDA kernel code
- **Multi-process support** for parallel generation
- **Comprehensive reward evaluation** based on:
  - CUDA syntax validity (40%)
  - Python syntax validity (20%)
  - Required function presence (20%)
  - CUDA implementation completeness (20%)
  - Optimization technique bonuses (up to 10%)

#### Reward Function Details
The reward function evaluates generated code based on:
1. **CUDA Kernel Markers**: Checks for `__global__`, `__device__`, kernel launches
2. **PyTorch Integration**: Verifies `load_inline` usage
3. **Required Components**: Ensures `ModelNew` class and `forward` method exist
4. **Code Quality**: Validates syntax and completeness
5. **Optimization Bonuses**: Rewards use of shared memory, thread synchronization, etc.

### 3. Loss Mask Generator

#### Custom Implementation
- **Location**: `/root/slime/slime/utils/mask_utils.py`
- **Method**: `gen_multi_turn_loss_mask_kernelbench`
- **Purpose**: Generates appropriate loss masks for kernel code training
- **Features**:
  - Masks system and user messages (no training)
  - Trains on entire assistant response (generated code)
  - Handles multi-turn conversations appropriately

### 4. Training Configuration

#### Training Script
- **Location**: `/root/slime/scripts/run-kernel-agent.sh`
- **Key Parameters**:
  ```bash
  --loss-mask-type kernelbench          # Use kernel-specific loss mask
  --rollout-max-response-len 12288      # Larger for CUDA code
  --rollout-temperature 0.8             # Higher for code diversity
  --n-samples-per-prompt 4              # Multiple attempts per problem
  --rollout-batch-size 64               # Batch size for generation
  --lr 5e-7                            # Lower learning rate for code
  ```

#### Model Configuration
- Base Model: DeepSeek-R1-Distill-Qwen-7B
- Sequence Length: 8192 tokens (increased for code)
- Attention Backend: Flash Attention
- Parallelism: TP=2, PP=1, CP=1

## Script Architecture

### Launch Scripts Overview

```
scripts/
├── run_kernel_agent.sh        # Master launcher (uses tmux)
├── kernel-agent-example.sh    # Training configuration script
└── start_rollout_buffer.sh    # Standalone buffer launcher
```

**Relationships:**
- `run_kernel_agent.sh` → Creates tmux session → Runs both components
  - Window 1: Executes `kernel-agent-example.sh`
  - Window 2: Executes `buffer.py` (after 30s delay)
- `kernel-agent-example.sh` → Standalone training script
- `start_rollout_buffer.sh` → Standalone buffer server

## Usage Guide

### Step 1: Prepare Environment

```bash
# Ensure you're in the Slime root directory
cd /root/slime

# Convert your data to the required format
python ./slime_plugins/rollout_buffer/tools/convert_kb_to_slime_format.py \
    --input_path /root/kb-rl-data/kb_hf_level_1.jsonl \
    --task_type kernelbench
```

### Step 2: Configure Rollout Buffer

The Rollout Buffer will automatically load the `kernel_generator.py` when it detects `task_type="kernelbench"`.

### Step 3: Start Training

**Recommended: Use the master launcher in an interactive terminal**
```bash
# This will create a tmux session with both components
./scripts/run_kernel_agent.sh
```

**Alternative: Manual setup in separate terminals**
```bash
# Terminal 1: Start Rollout Buffer
./scripts/start_rollout_buffer.sh

# Terminal 2: Start Training (after buffer is ready)
./scripts/kernel-agent-example.sh
```

### Step 4: Monitor Progress

Training progress can be monitored through:
1. **Console Output**: Real-time generation statistics
2. **WandB Dashboard**: Detailed metrics including:
   - Average reward
   - Success rate
   - Syntax validity rate
   - CUDA completeness rate
   - Optimization technique usage

## Customization Options

### 1. Reward Function Modification

Edit `kernel_utils.py` to adjust reward criteria:
```python
def evaluate_kernel_quality(code: str) -> float:
    # Modify scoring weights
    # Add new evaluation criteria
    # Adjust optimization bonuses
```

### 2. Generation Parameters

Modify in `kernel_generator.py`:
```python
SAMPLING_PARAMS = {
    "top_p": 0.95,          # Nucleus sampling
    "temperature": 0.8,      # Generation randomness
}
```

### 3. Data Processing

Implement custom functions in `kernel_generator.py`:
- `normalize_group_data`: Custom reward normalization
- `filter_item`: Additional data filtering
- `get_group_data_meta_info`: Custom statistics collection

## Evaluation Metrics

The system tracks several metrics for kernel code quality:

1. **Syntax Validity**: Both Python and CUDA syntax correctness
2. **Functional Completeness**: Presence of required methods and classes
3. **Optimization Quality**: Use of advanced CUDA features
4. **Generation Success Rate**: Percentage of valid kernels generated

## Troubleshooting

### Common Issues

1. **Memory Issues with Long Code**:
   - Reduce `rollout-batch-size`
   - Decrease `max-tokens-per-gpu`
   - Increase `sglang-mem-fraction-static`

2. **Low Reward Scores**:
   - Check if the model is generating proper code blocks
   - Verify the base model supports code generation
   - Adjust temperature for better diversity

3. **Generation Timeouts**:
   - Increase timeout settings in `query_kernel_generation`
   - Reduce `rollout-max-response-len`

### Debug Mode

Enable detailed logging by modifying the generator:
```python
# In kernel_generator.py
feedback = get_detailed_kernel_feedback(item)
print(f"Kernel evaluation: {feedback}")
```

## Best Practices

1. **Data Quality**: Ensure input problems are well-formatted and solvable
2. **Incremental Training**: Start with simpler kernels, progress to complex ones
3. **Regular Checkpointing**: Save models frequently (every 50 iterations)
4. **Reward Tuning**: Monitor and adjust reward function based on generated quality
5. **Batch Size Optimization**: Balance between GPU memory and training efficiency

## Future Enhancements

1. **Performance Benchmarking**: Integrate actual kernel execution timing
2. **Multi-GPU Kernel Support**: Extend to distributed CUDA kernels
3. **Template Learning**: Incorporate common optimization patterns
4. **Syntax-Aware Generation**: Use structured decoding for valid syntax
5. **Compilation Feedback**: Include nvcc compilation results in rewards

## References

- [Slime Agent Training Documentation](./agent_training.md)
- [Rollout Buffer Usage Documentation](./rollout_buffer_usage.md)
- [KernelBench Dataset](https://github.com/kernelbench/kernelbench)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review logs in the training output
3. Examine generated samples in the Rollout Buffer
4. Consult the Slime documentation for framework-specific issues