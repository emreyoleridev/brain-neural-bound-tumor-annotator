import os
import sys
import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

# Add project root to python path to resolve imports dynamically
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from training.models.unet import ResNet34UNet

class BrainTumorPredictor:
    """
    Production-ready inference pipeline for Brain Tumor Segmentation.
    Loads the trained model and processes incoming MRI scans.
    """
    def __init__(self, model_path: str, device: str = None):
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
            
        print(f"[INFO] Initializing Predictor on {self.device}...")
        
        # Initialize model architecture
        self.model = ResNet34UNet(out_classes=1)
        
        # Load trained weights
        if not os.path.exists(model_path):
            print(f"[INFO] Local weights not found at {model_path}. Trying to download from Hugging Face Hub...")
            try:
                from huggingface_hub import hf_hub_download
                # Kendi Hugging Face kullanıcı adınızı aşağıya yazın (emreyoleridev kısmına)
                hf_repo_id = "emreyoleridev/brain-neural-bound-tumor-model" 
                model_path = hf_hub_download(repo_id=hf_repo_id, filename="best_unet_model.pth")
                print(f"[INFO] Successfully downloaded weights to {model_path}")
            except Exception as e:
                raise FileNotFoundError(f"Weights not found locally and could not be downloaded from HF: {e}")
            
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval() # CRITICAL: Set to evaluation mode
        
        # Define the exact same preprocessing transforms used during validation
        self.transforms = A.Compose([
            A.Resize(256, 256),
            A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ])

    @torch.no_grad()
    def predict(self, image_path: str, threshold: float = 0.3) -> np.ndarray:
        """
        Takes an MRI image path, runs inference, and returns a binary mask.
        """
        # 1. Load image
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_size = (img.shape[1], img.shape[0]) # (Width, Height)
        
        # 2. Preprocess (Resize & Normalize)
        augmented = self.transforms(image=img)
        input_tensor = augmented['image'].unsqueeze(0).to(self.device) # Add batch dimension
        
        # 3. Model Inference
        logits = self.model(input_tensor)
        probs = torch.sigmoid(logits)
        
        # 4. Post-process (Apply threshold and convert back to numpy)
        # Squeeze removes batch and channel dims: [1, 1, 256, 256] -> [256, 256]
        mask = (probs.squeeze().cpu().numpy() >= threshold).astype(np.uint8)
        
        # 5. Resize mask back to original image dimensions for accurate overlay
        mask_resized = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)
        
        return mask_resized

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import glob
    import random
    
    print("[INFO] Testing Inference Pipeline...")
    
    weights_path = os.path.join(os.path.dirname(__file__), "weights", "best_unet_model.pth")
    predictor = BrainTumorPredictor(model_path=weights_path)
    
    all_images = glob.glob("data/**/*.tif", recursive=True)
    images_only = [f for f in all_images if not f.endswith("_mask.tif")]
    
    if not images_only:
        print("[ERROR] No images found to test.")
        sys.exit(1)
        
    print("[INFO] Searching for an MRI slice that actually contains a tumor...")
    test_image_path = None
    test_mask_path = None
    
    # Tümör içeren (siyah olmayan) bir maske bulana kadar rastgele seç
    while True:
        candidate_img = random.choice(images_only)
        candidate_mask = candidate_img.replace('.tif', '_mask.tif')
        
        # Maskeyi hızlıca oku ve içinde tümör (1/255) var mı bak
        mask_check = cv2.imread(candidate_mask, cv2.IMREAD_GRAYSCALE)
        if mask_check is not None and np.max(mask_check) > 0:
            test_image_path = candidate_img
            test_mask_path = candidate_mask
            break
            
    print(f"[INFO] Found tumor in: {test_image_path}")
    
    # Run Prediction
    predicted_mask = predictor.predict(test_image_path)
    
    # Load original image and ground truth for visualization
    orig_img = cv2.cvtColor(cv2.imread(test_image_path), cv2.COLOR_BGR2RGB)
    ground_truth = cv2.imread(test_mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Panel: Sadece Orijinal MR
    axes[0].imshow(orig_img)
    axes[0].set_title("Original MRI")
    axes[0].axis("off")
    
    # 2. Panel: Doktorun Çizimi (Maske)
    axes[1].imshow(ground_truth, cmap="gray")
    axes[1].set_title("Ground Truth (Doctor)")
    axes[1].axis("off")
    
    # 3. Panel: YAPAY ZEKA OVERLAY (MR Üstü Renklendirme)
    axes[2].imshow(orig_img) # Önce MR'ı çiziyoruz
    
    # Sadece tümörlü (1 olan) pikselleri alıp, geri kalanları (0 olanları) şeffaf yapıyoruz
    masked_prediction = np.ma.masked_where(predicted_mask == 0, predicted_mask)
    
    # Şeffaf tümör maskesini kırmızı/beyaz tonlarda (Reds) %50 saydamlıkla (alpha=0.5) MR'ın üstüne basıyoruz
    axes[2].imshow(masked_prediction, cmap="Reds", alpha=0.5) 
    
    axes[2].set_title("AI Prediction Overlay")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.show()