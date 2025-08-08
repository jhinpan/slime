from huggingface_hub import snapshot_download
import os

# Download Qwen3-8B model
model_id = "Qwen/Qwen3-8B"  
local_dir = "/root/Qwen3-8B"

print(f"Downloading {model_id} to {local_dir}...")
try:
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print(f"Download complete!")
except Exception as e:
    print(f"Error downloading model: {e}")
    print("Please check if the model exists on HuggingFace")