from huggingface_hub import HfApi

# Kendi bilgilerinizi buraya girin
HF_USERNAME = "emreyoleridev"
REPO_NAME = "medvision-unet"
REPO_ID = f"{HF_USERNAME}/{REPO_NAME}"
MODEL_PATH = "app/inference/weights/best_unet_model.pth"

api = HfApi()
print(f"Creating repo: {REPO_ID} ...")
try:
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True, private=False)
except Exception as e:
    print("Error creating repo:", e)

print(f"Uploading {MODEL_PATH} to {REPO_ID} ...")
api.upload_file(
    path_or_fileobj=MODEL_PATH,
    path_in_repo="best_unet_model.pth",
    repo_id=REPO_ID,
    repo_type="model"
)
print("Upload complete!")
