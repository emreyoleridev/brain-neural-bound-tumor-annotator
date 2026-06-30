import os
import sys
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from pathlib import Path

# Add project root to python path to resolve imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from training.datasets.brats_dataset import get_dataloaders
from training.models.unet import ResNet34UNet
from training.losses.dice_loss import DiceBCELoss
from training.metrics.segmentation_metrics import SegmentationMetrics

def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    """
    Trains the model for one epoch using Automatic Mixed Precision (AMP).
    """
    model.train()
    running_loss = 0.0
    
    # Progress bar
    pbar = tqdm(loader, desc="Training", leave=False)
    
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        
        # 1. Zero the gradients
        optimizer.zero_grad()
        
        # 2. Forward pass with AMP (Mixed Precision)
        with autocast():
            predictions = model(images)
            loss = criterion(predictions, masks)
            
        # 3. Backward pass and optimization using GradScaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
        
    return running_loss / len(loader)

@torch.no_grad()
def validate(model, loader, criterion, metric_calculator, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    
    # Metrics accumulators
    total_dice, total_iou = 0.0, 0.0
    
    pbar = tqdm(loader, desc="Validating", leave=False)
    
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        
        # Forward pass (no AMP needed for basic validation, but can be used)
        predictions = model(images)
        loss = criterion(predictions, masks)
        
        running_loss += loss.item()
        
        # Calculate metrics
        batch_metrics = metric_calculator(predictions, masks)
        total_dice += batch_metrics["dice"]
        total_iou += batch_metrics["iou"]
        
    avg_loss = running_loss / len(loader)
    avg_dice = total_dice / len(loader)
    avg_iou = total_iou / len(loader)
    
    return avg_loss, avg_dice, avg_iou

def main():
    # ---------------------------------------------------------
    # 1. CONFIGURATION & SETUP
    # ---------------------------------------------------------
    # Use GPU if available, else Apple Silicon (MPS), else CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"[INFO] Using Device: {device}")
    
    data_dir = "data"
    batch_size = 16 # Reduce to 8 or 4 if you run out of GPU memory (CUDA Out of Memory)
    epochs = 10     # Let's start with 10 epochs for this phase
    learning_rate = 1e-4
    
    # Create directory to save model weights
    save_dir = Path("app/inference/weights")
    save_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = save_dir / "best_unet_model.pth"

    # ---------------------------------------------------------
    # 2. INITIALIZE COMPONENTS
    # ---------------------------------------------------------
    print("[INFO] Initializing DataLoaders...")
    train_loader, val_loader = get_dataloaders(data_dir=data_dir, batch_size=batch_size)
    
    print("[INFO] Initializing Model, Loss, and Optimizer...")
    model = ResNet34UNet(out_classes=1).to(device)
    
    criterion = DiceBCELoss(bce_weight=0.5)
    metric_calculator = SegmentationMetrics(threshold=0.5)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    
    scaler = GradScaler() # For Mixed Precision
    
    best_val_loss = float('inf')

    # ---------------------------------------------------------
    # 3. TRAINING LOOP
    # ---------------------------------------------------------
    print("[INFO] Starting Training...")
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
        
        # Validate
        val_loss, val_dice, val_iou = validate(model, val_loader, criterion, metric_calculator, device)
        
        # Step the learning rate scheduler based on validation loss
        scheduler.step(val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f} | LR: {current_lr}")
        
        # Checkpoint: Save model if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"[*] Validation loss improved. Model saved to {best_model_path}")

    print(f"\n[SUCCESS] Training complete. Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()