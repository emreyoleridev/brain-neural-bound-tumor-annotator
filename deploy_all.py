import os
from huggingface_hub import HfApi

api = HfApi()
USERNAME = "emreyoleridev"
MODEL_REPO = f"{USERNAME}/medvision-unet"
SPACE_REPO = f"{USERNAME}/medvision-ai"
MODEL_PATH = "app/inference/weights/best_unet_model.pth"

# 1. Create and Upload to Model Repo
print("Creating Model Repo...")
try:
    api.create_repo(repo_id=MODEL_REPO, repo_type="model", exist_ok=True)
except Exception as e:
    print(f"Model repo creation error: {e}")

print("Uploading model weights...")
api.upload_file(
    path_or_fileobj=MODEL_PATH,
    path_in_repo="best_unet_model.pth",
    repo_id=MODEL_REPO,
    repo_type="model"
)
print("Model upload successful!")

# 2. Create and Upload to Space Repo
print("\nCreating Space Repo...")
try:
    api.create_repo(repo_id=SPACE_REPO, repo_type="space", space_sdk="docker", exist_ok=True)
except Exception as e:
    print(f"Space repo creation error: {e}")

print("Uploading space files...")
ignore_patterns = ["app/inference/weights/*", "venv/*", "*/__pycache__/*", ".git/*", "*.pt", "*.pth", "temp_input.tif"]

api.upload_folder(
    folder_path=".",
    repo_id=SPACE_REPO,
    repo_type="space",
    ignore_patterns=ignore_patterns
)

print("Space upload successful! Deployment Complete.")
