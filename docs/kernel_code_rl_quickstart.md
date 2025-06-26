# Kernel Code RL Quick Start Guide

## Prerequisites

- Slime framework configured and working
- KernelBench data available at `/root/kb-rl-data/kb_hf_level_1.jsonl`
- DeepSeek-R1-Distill-Qwen-7B model downloaded

## Quick Setup (5 Minutes)

### 1. Convert Data
```bash
cd /root/slime
python ./slime_plugins/rollout_buffer/tools/convert_kb_to_slime_format.py \
    --input_path /root/kb-rl-data/kb_hf_level_1.jsonl \
    --task_type kernelbench
```

### 2. Start Training

**Option A: Automatic (with tmux)**
```bash
# In an interactive terminal:
./scripts/run_kernel_agent.sh
```

**Option B: Manual (two terminals)**
```bash
# Terminal 1:
./scripts/start_rollout_buffer.sh

# Terminal 2 (after buffer starts):
./scripts/kernel-agent-example.sh
```

## Key Files Created

| File | Purpose |
|------|---------|
| `slime_plugins/rollout_buffer/generator/kernel_generator.py` | Main kernel code generator |
| `slime_plugins/rollout_buffer/generator/reward_utils/kernel_utils.py` | Reward calculation for CUDA code |
| `slime/utils/mask_utils.py` (modified) | Added kernelbench loss mask support |
| `scripts/run-kernel-agent.sh` | Training launch script |

## Important Parameters

```bash
--rollout-task-type kernelbench      # Activates kernel generator
--loss-mask-type distill_qwen        # Uses appropriate loss mask for the model  
--rollout-max-response-len 12288     # Supports long CUDA code
--n-samples-per-prompt 4             # Generate 4 variants per problem
```

## Monitoring

Watch for these metrics in the output:
- **avg_reward**: Should increase over time (target: >0.7)
- **syntax_valid_rate**: Percentage of syntactically correct code
- **cuda_complete_rate**: Percentage with complete CUDA implementations
- **successful_generations**: Count of high-quality kernels

## Quick Debugging

### Check Generated Samples
```bash
# In another terminal, query the rollout buffer
curl http://localhost:8889/get_rollout_data
```

### Verify Reward Calculation
```python
from slime_plugins.rollout_buffer.generator.reward_utils.kernel_utils import get_kernel_code_reward

# Test with a sample
item = {"messages": [{"role": "assistant", "content": "your_generated_code"}]}
reward = get_kernel_code_reward(item)
print(f"Reward: {reward}")
```

## Common Issues & Fixes

| Issue | Fix |
|-------|-----|
| Low rewards (<0.3) | Increase temperature to 0.9 |
| OOM errors | Reduce rollout-batch-size to 32 |
| Slow generation | Reduce rollout-max-response-len to 8192 |
| No CUDA kernels | Check if model understands the task format |

## Next Steps

1. Monitor training for 100-200 iterations
2. Evaluate generated kernels quality
3. Fine-tune reward function if needed
4. Scale up with more data or larger batches

For detailed information, see [Full Implementation Guide](./kernel_code_rl_implementation.md)