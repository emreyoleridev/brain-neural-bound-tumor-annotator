import torch
from typing import Dict

class SegmentationMetrics:
    """
    Computes standard medical image segmentation metrics:
    Dice Score, IoU, Precision, and Recall.
    """
    def __init__(self, threshold: float = 0.5, smooth: float = 1e-6):
        self.threshold = threshold
        self.smooth = smooth

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Calculates metrics for a batch of predictions and targets.
        
        Args:
            logits: Raw model outputs (before sigmoid) [B, C, H, W]
            targets: Ground truth binary masks [B, C, H, W]
            
        Returns:
            Dictionary containing the average Dice, IoU, Precision, and Recall.
        """
        # 1. Convert logits to probabilities
        probs = torch.sigmoid(logits)
        
        # 2. Apply threshold to get hard binary predictions (0 or 1)
        preds = (probs >= self.threshold).float()
        
        # 3. Flatten tensors
        preds = preds.view(-1)
        targets = targets.view(-1)
        
        # 4. Calculate True Positives, False Positives, False Negatives
        # Using dot products/sums for highly efficient GPU computation
        tp = (preds * targets).sum()
        fp = (preds * (1 - targets)).sum()
        fn = ((1 - preds) * targets).sum()
        
        # 5. Compute Metrics
        dice = (2.0 * tp + self.smooth) / (2.0 * tp + fp + fn + self.smooth)
        iou = (tp + self.smooth) / (tp + fp + fn + self.smooth)
        
        # Precision: Out of all predicted positives, how many are real?
        precision = (tp + self.smooth) / (tp + fp + self.smooth)
        
        # Recall: Out of all real positives, how many did we find?
        recall = (tp + self.smooth) / (tp + fn + self.smooth)
        
        return {
            "dice": dice.item(),
            "iou": iou.item(),
            "precision": precision.item(),
            "recall": recall.item()
        }

if __name__ == "__main__":
    # ---------------------------------------------------------
    # METRICS SANITY CHECK
    # ---------------------------------------------------------
    print("[INFO] Testing Segmentation Metrics...")
    
    # Simulate a target mask with a 2x2 tumor
    targets = torch.zeros(1, 1, 4, 4)
    targets[0, 0, 1:3, 1:3] = 1.0  
    
    # Simulate a prediction that misses 1 pixel of the tumor and hallucinates 1 pixel outside
    logits = torch.zeros(1, 1, 4, 4) - 10.0 # Background (Sigmoid gets ~0)
    logits[0, 0, 1:3, 1:2] = 10.0 # Tumor prediction (TP) - missing 1 col
    logits[0, 0, 0, 0] = 10.0     # False Positive
    
    metric_calculator = SegmentationMetrics(threshold=0.5)
    metrics = metric_calculator(logits, targets)
    
    print(f"Ground Truth Tumor Pixels: {int(targets.sum().item())}")
    print("Metrics evaluated:")
    for k, v in metrics.items():
        print(f" - {k.capitalize()}: {v:.4f}")
        
    print("[SUCCESS] Metrics module is ready.")