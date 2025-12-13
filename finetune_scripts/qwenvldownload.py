import os
from huggingface_hub import snapshot_download

hf_model_id = "Qwen/Qwen2-VL-7B-Instruct"
# Optional: Enable faster downloads

# Download model locally to a directory named "Qwen2-VL-7B-Instruct-local"
local_directory = "./Qwen2-VL-7B-Instruct-local"
snapshot_download(repo_id=hf_model_id, local_dir=local_directory)