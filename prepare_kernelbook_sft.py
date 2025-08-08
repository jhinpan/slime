"""
Prepare GPUMode/KernelBook dataset for SFT training with Triton kernel generation prompt
"""

from datasets import load_dataset
import json
from pathlib import Path
from tqdm import tqdm

# System prompt for Triton kernel generation
SYSTEM_PROMPT = """You are an expert in writing Triton kernels for efficient GPU programming.

You are a helpful assistant that can generate code to solve a problem.
You write custom Triton kernels to replace the pytorch operators in the given architecture to get speedups.

You have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom Triton kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.

    
Here's an example to show you the syntax of inline embedding custom Triton kernels in torch: The example given architecture is: 

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        return a + b

def get_inputs():
    # randomly generate input tensors based on the model architecture
    a = torch.randn(1, 128).cuda()
    b = torch.randn(1, 128).cuda()
    return [a, b]

def get_init_inputs():
    # randomly generate tensors required for initialization based on the model architecture
    return []
``` 

The example new arch with custom Triton kernels looks like this: 
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,  # Pointer to first input
    y_ptr,  # Pointer to second input
    out_ptr,  # Pointer to output
    n_elements,  # Total number of elements in input/output
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    # Perform the elementwise addition
    out = x + y
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_add(x: torch.Tensor, y: torch.Tensor):
    \"\"\"
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    \"\"\"
    assert x.is_cuda and y.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    y = y.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        # Instead of "return a + b", call our Triton-based addition
        return triton_add(a, b)

def get_inputs():
    # randomly generate input tensors based on the model architecture
    a = torch.randn(4096).cuda()
    b = torch.randn(4096).cuda()
    return [a, b]

def get_init_inputs():
    # randomly generate tensors required for initialization based on the model architecture
    return []
```"""

USER_PROMPT_PREFIX = """You are given the following architecture: 
```python
{input_code}
```

Optimize the architecture named Model with custom Triton kernels! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code!"""

def process_dataset():
    print("Loading KernelBook dataset...")
    dataset = load_dataset("GPUMODE/KernelBook", split="train")
    
    print(f"Dataset size: {len(dataset)} samples")
    
    # Prepare output file
    output_file = Path("/root/kernelbook_sft.jsonl")
    
    processed_count = 0
    skipped_count = 0
    
    with open(output_file, "w") as f:
        for item in tqdm(dataset, desc="Processing samples"):
            try:
                # Extract input and output code - using correct field names
                input_code = item.get("python_code", "")
                output_code = item.get("triton_code", "")
                
                if not input_code or not output_code:
                    skipped_count += 1
                    continue
                
                # Create the conversation format for SFT
                messages = [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user", 
                        "content": USER_PROMPT_PREFIX.format(input_code=input_code)
                    },
                    {
                        "role": "assistant",
                        "content": f"```python\n{output_code}\n```"
                    }
                ]
                
                # Write as JSONL for slime SFT training
                json_line = json.dumps({"messages": messages})
                f.write(json_line + "\n")
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing item: {e}")
                skipped_count += 1
                continue
    
    print(f"\nProcessing complete!")
    print(f"Processed: {processed_count} samples")
    print(f"Skipped: {skipped_count} samples")
    print(f"Output saved to: {output_file}")
    
    # Also create a smaller evaluation set (first 100 samples)
    eval_file = Path("/root/kernelbook_eval.jsonl")
    with open(output_file, "r") as f_in, open(eval_file, "w") as f_out:
        for i, line in enumerate(f_in):
            if i >= 100:
                break
            f_out.write(line)
    
    print(f"Evaluation set (100 samples) saved to: {eval_file}")

if __name__ == "__main__":
    process_dataset()