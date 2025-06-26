import argparse
import json
from pathlib import Path


def convert_kb_to_slime_format(input_path, task_type="code", output_path=None):
    """
    Convert KernelBench training data format to Slime expected format.
    
    Args:
        input_path: Path to the input JSONL file (kb_hf_level_1.jsonl format)
        task_type: Task type prefix for instance_id (default: "code")
        output_path: Optional path to output file. If None, will create a _processed.jsonl file
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = str(input_path).replace(".jsonl", "_processed.jsonl")
    
    processed = []
    
    # Read and convert each line
    with open(input_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            item = json.loads(line)
            
            # Create the prompt in OpenAI message format
            messages = []
            
            # Add system message if instruction exists
            if item.get("instruction"):
                messages.append({
                    "role": "system",
                    "content": item["instruction"]
                })
            
            # Add user message with the input
            if item.get("input"):
                messages.append({
                    "role": "user", 
                    "content": item["input"]
                })
            
            # Create converted item
            converted_item = {
                "prompt": messages,
                "label": item.get("output", ""),  # Use output as label
                "instance_id": f"{task_type}_{item.get('problem_id', idx)}"  # Use problem_id or index
            }
            
            processed.append(converted_item)
    
    # Save to new jsonl file
    with open(output_path, "w", encoding="utf-8") as f:
        for item in processed:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"✅ Converted {len(processed)} items from KernelBench format to Slime format.")
    print(f"📁 Saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert KernelBench training data to Slime format"
    )
    parser.add_argument(
        "--input_path", 
        type=str, 
        required=True,
        help="Path to input JSONL file in KernelBench format"
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default="code",
        help="Task type prefix for instance_id (default: code)"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default=None, 
        help="Optional path to output file (default: input_path with _processed suffix)"
    )
    
    args = parser.parse_args()
    convert_kb_to_slime_format(args.input_path, args.task_type, args.output_path)


if __name__ == "__main__":
    main()