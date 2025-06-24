# Agent Rollout Walkthrough for Slime Framework

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Setup and Configuration](#setup-and-configuration)
4. [Implementation Guide](#implementation-guide)
5. [Code Examples](#code-examples)
6. [Running Agent Training](#running-agent-training)
7. [Advanced Topics](#advanced-topics)

## Overview

The Slime framework's agent rollout system enables fully asynchronous reinforcement learning training by decoupling trajectory generation from model training. This design allows you to integrate any agent framework while leveraging Slime's high-performance training infrastructure.

### Key Benefits
- **Framework Agnostic**: Use any agent framework (LangChain, AutoGPT, custom implementations)
- **Asynchronous Operation**: Training and rollout generation happen concurrently
- **Scalable**: Support for 1000+ parallel rollout processes
- **Resume Support**: Automatic handling of interrupted training

## Architecture

### System Components

```
┌─────────────────────┐         HTTP API          ┌──────────────────────┐
│   Slime Training    │ ←───────────────────────→ │   Rollout Buffer     │
│   (Megatron-LM)     │                           │  (External Service)  │
└──────────┬──────────┘                           └──────────┬───────────┘
           │                                                 │
           ↓                                                 ↓
┌─────────────────────┐                           ┌──────────────────────┐
│   SGLang Server     │ ←── HTTP Inference ─────→ │   Agent Framework    │
│   (Model Serving)   │                           │  (Your Custom Code)  │
└─────────────────────┘                           └──────────────────────┘
```

### Data Flow

1. **Initialization**: Slime sends configuration to Rollout Buffer
2. **Generation**: Rollout Buffer uses SGLang server to generate trajectories
3. **Collection**: Slime polls for completed trajectories
4. **Training**: Collected data is used for model updates
5. **Weight Sync**: Updated weights are pushed to SGLang periodically

## Setup and Configuration

### 1. Environment Setup

First, ensure you have the Slime environment ready:

```bash
# Using the official Docker image
docker run --rm --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -it zhuzilin/slime:latest /bin/bash

cd /root/slime
pip install -e .
```

### 2. Key Configuration Files

#### `scripts/agent-example.sh` - Main Configuration
```bash
# Model architecture (DeepSeek-R1-Distill-Qwen-7B example)
MODEL_ARGS=(
   --swiglu
   --num-layers 28
   --hidden-size 3584
   --ffn-hidden-size 18944
   --num-attention-heads 28
   --group-query-attention
   --num-query-groups 4
   --max-position-embeddings 131072
   --seq-length 4096
   --vocab-size 152064
)

# Rollout configuration
ROLLOUT_ARGS=(
   --rollout-function-path slime.rollout.agent_rollout.generate_rollout
   --rm-type deepscaler
   --prompt-data ${PROMPT_DATA}
   --num-rollout 3000
   --rollout-batch-size 128
   --rollout-max-response-len 8192
   --rollout-temperature 0.8
   --n-samples-per-prompt 8
)

# Agent-specific settings
--agent-rollout-buffer-url http://${MASTER_ADDR}:8889
--rollout-num-process 1024
--loss-mask-type distill_qwen
```

#### `scripts/run_agent.sh` - Launch Script
```bash
#!/bin/bash
SESSION_NAME="slime_run"
WINDOW_1="slime"
WINDOW_2="buffer"

# Launch Slime training in first window
tmux new-session -d -s $SESSION_NAME -n $WINDOW_1
tmux send-keys -t ${SESSION_NAME}:${WINDOW_1} "bash ./scripts/agent-example.sh" C-m

# Launch Rollout Buffer in second window
tmux new-window -t $SESSION_NAME -n $WINDOW_2
tmux send-keys -t ${SESSION_NAME}:${WINDOW_2} "sleep 30 && cd slime_plugins/rollout_buffer && python buffer.py" C-m

tmux attach-session -t $SESSION_NAME
```

## Implementation Guide

### Step 1: Create Your Generator

Create a new file in `slime_plugins/rollout_buffer/generator/your_task_generator.py`:

```python
import copy
import json
import random
import time
from openai import OpenAI
from generator.reward_utils.math_utils import get_rule_based_math_reward

# REQUIRED: Define your task type
TASK_TYPE = "your_custom_task"

# REQUIRED: Main rollout function
def run_rollout(data: dict):
    """
    Entry point for your custom rollout logic.
    
    Args:
        data: Configuration from Slime containing:
            - remote_engine_url: SGLang server URL
            - remote_buffer_url: Buffer server URL  
            - input_file: Path to input data
            - num_repeat_per_sample: Samples per prompt
            - sampling_params: Model sampling config
            - num_process: Parallel processes
            - skip_instance_ids: IDs to skip (for resume)
    """
    print(f"Starting {TASK_TYPE} rollout with data: {data}")
    
    # Initialize your generator
    generator = BaseGenerator(
        data["remote_engine_url"],
        data["remote_buffer_url"],
        num_repeat_per_sample=int(data["num_repeat_per_sample"]),
        queue_size=1000000,
        max_tokens=int(data["sampling_params"]["max_tokens"]),
        num_process=int(data.get("num_process", 100)),
        task_type=data["task_type"],
        skip_instance_ids=data.get("skip_instance_ids", None),
    )
    
    # Define your rollout and reward functions
    rollout_func = your_custom_rollout_function
    reward_func = your_custom_reward_function
    
    # Start generation
    generator.entry(
        data["input_file"], 
        rollout_func, 
        reward_func, 
        int(data.get("num_epoch", 1))
    )

# Example rollout function (from base_generator.py)
def query_single_turn(client, messages, sampling_params, tools=None):
    """Generate a single response from the model."""
    base_payload = {
        "messages": messages,
        **sampling_params,
        "model": "custom",
        "stream": False,
        "seed": random.randint(1, 10000000),
        "tools": tools,
    }
    
    text = None
    accumulated_tokens = 0
    
    for attempt in range(6):
        try:
            current_payload = copy.deepcopy(base_payload)
            
            # Handle continuation for long responses
            if text is not None:
                current_messages = copy.deepcopy(messages)
                current_messages.append({"role": "assistant", "content": text})
                current_payload["messages"] = current_messages
                
                if "max_tokens" in sampling_params:
                    current_payload["max_tokens"] = max(
                        0, 
                        sampling_params["max_tokens"] - accumulated_tokens
                    )
                
                current_payload["extra_body"] = {"continue_final_message": True}
            
            response = client.chat.completions.create(**current_payload)
            
            if len(response.choices) > 0:
                if response.choices[0].finish_reason == "abort":
                    # Handle partial generation
                    accumulated_tokens += response.usage.completion_tokens
                    text = (text or "") + response.choices[0].message.content
                    time.sleep(10)
                    continue
                    
                # Success - combine text
                if text is None:
                    text = response.choices[0].message.content
                elif response.choices[0].message.content is not None:
                    text += response.choices[0].message.content
                break
                
        except Exception as e:
            print(f"Query failed: {e}")
            continue
    
    # Update messages with final response
    messages.append({"role": "assistant", "content": text})
    return messages
```

### Step 2: Implement Optional Functions

Add these optional functions to customize behavior:

```python
def normalize_group_data(group_data):
    """
    Normalize rewards within a group.
    Default: Normalizes valid rewards and scales by group_size/valid_size
    """
    valid_rewards = [item["reward"] for item in group_data if item["reward"] > -1]
    
    if not valid_rewards:
        return group_data
    
    mean_reward = sum(valid_rewards) / len(valid_rewards)
    std_reward = (sum((r - mean_reward) ** 2 for r in valid_rewards) / len(valid_rewards)) ** 0.5
    
    for item in group_data:
        if item["reward"] > -1:
            item["reward"] = (item["reward"] - mean_reward) / (std_reward + 1e-8)
            # Scale to maintain total reward
            item["reward"] *= len(group_data) / len(valid_rewards)
    
    return group_data

def is_valid_group(instance_id, group_data):
    """
    Determine if a group is valid for training.
    Returns: (is_valid, is_finished)
    """
    valid_count = sum(1 for item in group_data if item.get("reward", -1) > -1)
    total_count = len(group_data)
    
    # Group is finished if we have enough samples
    is_finished = total_count >= 8  # or your target group size
    
    # Group is valid if >70% samples are valid
    is_valid = is_finished and (valid_count / total_count >= 0.7)
    
    return is_valid, is_finished

def filter_item(item):
    """Filter individual items within valid groups."""
    # Keep items with valid rewards
    return item.get("reward", -1) > -1

def get_group_data_meta_info(all_group_data):
    """Collect statistics for monitoring."""
    total_samples = sum(len(group) for group in all_group_data.values())
    valid_samples = sum(
        sum(1 for item in group if item.get("reward", -1) > -1)
        for group in all_group_data.values()
    )
    
    all_rewards = [
        item["reward"] 
        for group in all_group_data.values() 
        for item in group 
        if item.get("reward", -1) > -1
    ]
    
    return {
        "total_samples": total_samples,
        "valid_samples": valid_samples,
        "avg_reward": sum(all_rewards) / len(all_rewards) if all_rewards else 0,
        "max_reward": max(all_rewards) if all_rewards else 0,
        "min_reward": min(all_rewards) if all_rewards else 0,
    }
```

### Step 3: Configure Loss Mask (if needed)

For custom conversation formats, add to `slime/utils/mask_utils.py`:

```python
class YourTaskLossMaskGenerator(MultiTurnLossMaskGenerator):
    def get_loss_mask(self, messages: List[Dict]) -> List[int]:
        if self.tokenizer_type == "your_task":
            # Custom logic for your conversation format
            token_ids = []
            loss_mask = []
            
            for message in messages:
                role = message["role"]
                content = message["content"]
                
                # Tokenize the message
                msg_tokens = self.tokenizer.encode(content)
                token_ids.extend(msg_tokens)
                
                # Only train on assistant responses
                if role == "assistant":
                    loss_mask.extend([1] * len(msg_tokens))
                else:
                    loss_mask.extend([0] * len(msg_tokens))
            
            return token_ids, loss_mask
```

## Code Examples

### Example 1: Math Task Implementation (from base_generator.py)

```python
# Worker process for parallel generation
def worker_process(task_queue, done_queue, rollout_func, reward_func, client, sampling_params):
    for line in iter(task_queue.get, "STOP"):
        if isinstance(line, str):
            item = json.loads(line)
        else:
            item = line
        
        group_data = []
        messages_set = set()
        
        # Generate multiple samples per prompt
        for _ in range(sampling_params.get("n", 1)):
            messages = copy.deepcopy(item["prompt"])
            
            # Run rollout
            generated_messages = rollout_func(client, messages, sampling_params)
            
            # Calculate reward
            reward = reward_func(generated_messages)
            
            # Create result item
            result_item = {
                "uid": str(uuid.uuid4()),
                "instance_id": item["instance_id"],
                "messages": generated_messages,
                "reward": reward,
                "raw_reward": reward,  # Store original reward
                "extra_info": {
                    "timestamp": time.time(),
                    "label": item.get("label", None)
                }
            }
            
            group_data.append(result_item)
        
        done_queue.put(group_data)
```

### Example 2: Agent Rollout Communication (from agent_rollout.py)

```python
async def generate_agent_rollout(args, rollout_id: int, data_buffer: Buffer, evaluation: bool = False):
    """Main entry point for agent rollout in Slime."""
    global TOKENIZER, START_ROLLOUT
    
    # Initialize tokenizer once
    if TOKENIZER is None:
        TOKENIZER = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    
    # Start rollout on first iteration
    if START_ROLLOUT:
        metadata = {"skip_instance_ids": []}
        if rollout_id > 0:
            # Get completed instance IDs for resume
            metadata = data_buffer.get_metadata()
        
        start_rollout(args.agent_rollout_buffer_url, args, metadata)
        START_ROLLOUT = False
    
    # Collect results
    results = []
    all_meta_info = []
    retry_count = 0
    
    while len(results) < args.rollout_batch_size * args.n_samples_per_prompt:
        await asyncio.sleep(5)  # Poll interval
        
        try:
            data, meta_info = await get_rollout_data(
                api_base_url=args.agent_rollout_buffer_url
            )
            
            if data:
                results.extend(data)
                if meta_info:
                    all_meta_info.append(meta_info)
                retry_count = 0
            else:
                retry_count += 1
                
        except Exception as e:
            print(f"Error fetching data: {e}")
            retry_count += 1
            
        # Check retry limit
        if args.fetch_trajectory_retry_times != -1 and retry_count > args.fetch_trajectory_retry_times:
            raise RuntimeError(f"Failed to fetch data after {retry_count} retries")
    
    # Convert to Slime format
    samples = []
    for record in results:
        oai_messages = record["messages"]
        
        # Generate loss mask
        mask_generator = MultiTurnLossMaskGenerator(
            TOKENIZER, 
            tokenizer_type=args.loss_mask_type
        )
        token_ids, loss_mask = mask_generator.get_loss_mask(oai_messages)
        response_length = mask_generator.get_response_lengths([loss_mask])[0]
        
        sample = Sample(
            index=record["instance_id"],
            prompt=record["uid"],
            tokens=token_ids,
            response_length=response_length,
            reward=record["reward"],
            truncated=False,
            loss_mask=loss_mask[-response_length:],
            metadata={
                **record["extra_info"], 
                "raw_reward": record["raw_reward"]
            }
        )
        samples.append(sample)
    
    # Add to buffer and log
    data_buffer.extend(samples)
    log_raw_info(args, all_meta_info, rollout_id)
    
    return samples
```

### Example 3: Async Training Loop (from train_agent_async.py)

```python
def train(args):
    # Create placement groups for GPUs
    pgs = create_placement_groups(args)
    
    # Initialize actor model and rollout generator
    actor_model = create_actor_group(args, pgs["actor"])
    rollout_generator = create_rollout_group(args, pgs["rollout"])
    
    # Initialize and sync weights
    ray.get(actor_model.async_init(args, role="actor", with_ref=args.use_kl_loss))
    ray.get(actor_model.async_init_weight_update_connections(rollout_generator))
    ray.get(actor_model.async_update_weights())
    
    # Start first generation
    generation_handles = rollout_generator.async_generate(args.start_rollout_id)
    
    # Async training loop
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # Wait for generation to complete
        ray.get(generation_handles)
        
        # Fetch data from rollout buffer
        actor_model.get_rollout_data(rollout_id)
        
        # Train asynchronously
        actor_handles = actor_model.async_train(rollout_id, with_data_fetching=False)
        
        # Wait for training to complete
        ray.get(actor_handles)
        
        # Update weights periodically
        if (rollout_id + 1) % args.update_rollout_weights_interval == 0:
            ray.get(actor_model.async_update_weights())
        
        # Start next generation (overlaps with training)
        generation_handles = rollout_generator.async_generate(rollout_id + 1)
        
        # Save checkpoint periodically
        if (rollout_id + 1) % args.save_interval == 0:
            ray.get(actor_model.async_save_model(rollout_id))
```

## Running Agent Training

### Step 1: Prepare Your Data

Create input file with instance IDs:
```bash
python ./slime_plugins/rollout_buffer/tools/assign_instance_id.py \
    --input_path /path/to/your_data.jsonl
```

Input format example:
```jsonl
{"instance_id": "task_001", "prompt": [{"role": "user", "content": "Your task prompt here"}]}
{"instance_id": "task_002", "prompt": [{"role": "user", "content": "Another task prompt"}]}
```

### Step 2: Configure Your Training Script

Copy and modify `scripts/agent-example.sh`:
```bash
# Key parameters to modify
export HF_MODEL_PATH=/path/to/your/model
export PROMPT_DATA=/path/to/your/data_with_ids.jsonl
export TASK_TYPE=your_custom_task

ROLLOUT_ARGS=(
   --rollout-function-path slime.rollout.agent_rollout.generate_rollout
   --prompt-data ${PROMPT_DATA}
   --rollout-batch-size 128      # Samples per training batch
   --n-samples-per-prompt 8      # Generations per prompt
   --rollout-temperature 0.8     # Sampling temperature
   --rollout-max-response-len 8192
   --loss-mask-type your_task    # Your custom mask type
)

# Agent-specific settings
--agent-rollout-buffer-url http://${MASTER_ADDR}:8889
--rollout-num-process 1024    # Parallel generation processes
--update-rollout-weights-interval 1  # Sync weights every iteration
```

### Step 3: Launch Training

```bash
# Make script executable
chmod +x ./scripts/run_agent.sh

# Launch training (opens tmux session)
./scripts/run_agent.sh
```

### Step 4: Monitor Progress

In the tmux session:
- Window 1 (slime): Shows training progress, loss, rewards
- Window 2 (buffer): Shows rollout generation status

Switch windows: `Ctrl+B` then `0` or `1`

## Advanced Topics

### 1. Resume Training

The system automatically handles resume:
```python
# In your generator, skip completed instances
if self.skip_instance_ids and instance_id in self.skip_instance_ids:
    continue
```

### 2. Custom Reward Functions

Implement domain-specific rewards:
```python
def custom_reward_function(messages):
    """Calculate reward based on your criteria."""
    last_response = messages[-1]["content"]
    
    # Example: Length-based reward
    length_reward = min(len(last_response) / 1000, 1.0)
    
    # Example: Quality checks
    quality_reward = 0
    if "correct_answer" in last_response:
        quality_reward = 1.0
    
    return length_reward * 0.3 + quality_reward * 0.7
```

### 3. Handling Timeouts

Configure timeout parameters in buffer:
```python
# In your generator configuration
group_timeout_seconds = 300  # 5 minutes
min_timeout_group_size_ratio = 0.7  # Accept if 70% complete
```

### 4. Debugging Tips

1. **Check SGLang Connection**:
   ```bash
   curl -X POST http://localhost:30000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "custom", "messages": [{"role": "user", "content": "test"}]}'
   ```

2. **Monitor Buffer Status**:
   ```bash
   curl http://localhost:8889/stats
   ```

3. **Enable Debug Logging**:
   ```bash
   export PYTHONUNBUFFERED=1
   export SGLANG_LOG_LEVEL=debug
   ```

### 5. Performance Optimization

1. **Batch Size Tuning**:
   - Larger `rollout-batch-size`: More stable training
   - Smaller batches: Faster iteration

2. **Process Count**:
   - Set `rollout-num-process` based on task complexity
   - Monitor CPU usage and adjust

3. **Weight Update Frequency**:
   - `--update-rollout-weights-interval 1`: Every iteration (slower)
   - Higher values: Less frequent updates (faster generation)

## Troubleshooting

### Common Issues

1. **"Failed to fetch data" errors**:
   - Check Rollout Buffer is running: `ps aux | grep buffer.py`
   - Verify URL is correct in configuration

2. **Out of memory**:
   - Reduce `rollout-batch-size`
   - Decrease `max_buffer_size` in generator

3. **Slow generation**:
   - Increase `rollout-num-process`
   - Check SGLang server load

4. **Resume not working**:
   - Ensure instance IDs are consistent
   - Check `skip_instance_ids` is being passed

### Getting Help

- Check logs in both tmux windows
- Review `/tmp/ray/session_*/logs/` for detailed errors
- Enable debug mode in your generator for more output

## Conclusion

The Slime agent rollout system provides a flexible, scalable foundation for RL training with any agent framework. By following this guide and adapting the examples to your use case, you can implement sophisticated agent training pipelines while leveraging Slime's high-performance infrastructure.

Key takeaways:
- Keep rollout logic separate in the Rollout Buffer
- Use async operations for maximum efficiency  
- Monitor and tune based on your specific requirements
- Leverage resume support for long-running experiments