from datasets import load_dataset

# Load a single sample to check structure
dataset = load_dataset("GPUMODE/KernelBook", split="train", streaming=True)

# Get first sample
sample = next(iter(dataset))

print("Dataset fields:")
for key in sample.keys():
    print(f"  - {key}")
    
print("\nFirst sample content:")
for key, value in sample.items():
    print(f"\n{key}:")
    if isinstance(value, str):
        print(value[:500] if len(value) > 500 else value)
    else:
        print(value)