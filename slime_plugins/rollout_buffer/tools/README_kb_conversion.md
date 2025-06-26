# KernelBench to Slime Format Conversion Tool

This tool converts KernelBench training data format to the format expected by Slime for agent training.

## Usage

### Basic Usage

```bash
python ./slime_plugins/rollout_buffer/tools/convert_kb_to_slime_format.py \
    --input_path /root/kb-rl-data/kb_hf_level_1.jsonl \
    --task_type kernelbench
```

### Command Line Arguments

- `--input_path`: Path to input JSONL file in KernelBench format (required)
- `--task_type`: Task type prefix for instance_id (default: "code")
- `--output_path`: Optional path to output file (default: input_path with _processed suffix)

## Format Conversion Details

### Input Format (KernelBench)
```json
{
    "instruction": "You are an expert in writing CUDA Kernels...",
    "input": "You write custom CUDA kernels to replace...",
    "output": "",
    "problem_id": 100
}
```

### Output Format (Slime)
```json
{
    "prompt": [
        {
            "role": "system",
            "content": "You are an expert in writing CUDA Kernels..."
        },
        {
            "role": "user",
            "content": "You write custom CUDA kernels to replace..."
        }
    ],
    "label": "",
    "instance_id": "kernelbench_100"
}
```

## Integration with Slime Training

After conversion, you can use the processed file in the Slime training script:

```bash
export PROMPT_DATA=/root/kb-rl-data/kb_hf_level_1_processed.jsonl

# Then run your training script
./scripts/run_agent.sh
```

Make sure to update the `agent-example.sh` script to use the converted data path.