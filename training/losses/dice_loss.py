import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Computes the Dice Loss for binary segmentation.
    Includes a sigmoid activation to map model outputs to probabilities [0,1].
    """
    def __init__(self, smooth: float = 1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 1. Apply sigmoid to model logits to get probabilities
        probs = torch.sigmoid(logits)
        
        # 2. Flatten both tensors from [B, C, H, W] to [B, C*H*W]
        # This makes it easier to compute the intersection and union across the whole batch
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        # 3. Compute Intersection and Union
        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()
        
        # 4. Compute Dice Score
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # 5. Return Dice Loss (1 - Dice Score)
        return 1.0 - dice_score

class DiceBCELoss(nn.Module):
    """
    Combined Dice Loss and Binary Cross Entropy (BCE) Loss.
    Combines the pixel-wise stability of BCE with the class-imbalance robustness of Dice.
    """
    def __init__(self, smooth: float = 1e-6, bce_weight: float = 0.5):
        super(DiceBCELoss, self).__init__()
        self.dice_loss = DiceLoss(smooth=smooth)
        self.bce_weight = bce_weight
        self.dice_weight = 1.0 - bce_weight
        
        # BCEWithLogitsLoss is numerically more stable than applying Sigmoid followed by BCELoss
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Calculate individual losses
        dice = self.dice_loss(logits, targets)
        bce = self.bce_loss(logits, targets)
        
        # Combine them
        combined_loss = (self.bce_weight * bce) + (self.dice_weight * dice)
        
        return combined_loss

if __name__ == "__main__":
    # ---------------------------------------------------------
    # LOSS FUNCTIONS SANITY CHECK
    # ---------------------------------------------------------
    print("[INFO] Testing Loss Functions...")
    
    # 1. Simulate perfect prediction
    dummy_logits_perfect = torch.ones(2, 1, 256, 256) * 10.0  # High values -> sigmoid -> ~1.0
    dummy_targets = torch.ones(2, 1, 256, 256)
    
    # 2. Simulate terrible prediction
    dummy_logits_terrible = torch.ones(2, 1, 256, 256) * -10.0 # Low values -> sigmoid -> ~0.0
    
    criterion = DiceBCELoss()
    
    loss_perfect = criterion(dummy_logits_perfect, dummy_targets)
    loss_terrible = criterion(dummy_logits_terrible, dummy_targets)
    
    print(f"Loss with perfect prediction: {loss_perfect.item():.4f} (Expected near 0)")
    print(f"Loss with terrible prediction: {loss_terrible.item():.4f} (Expected much higher than 0)")
    print("[SUCCESS] Loss functions are ready.")