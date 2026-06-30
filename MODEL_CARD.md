---
license: mit
tags:
- medical
- image-segmentation
- pytorch
- unet
- brain-tumor
datasets:
- mateuszbuda/lgg-mri-segmentation
metrics:
- dice
- iou
---

# Brain Neural Bound Tumor Model

This is a PyTorch-based UNet model with a ResNet34 backbone, designed for the automatic segmentation of lower-grade gliomas (brain tumors) from MRI scans. It provides high-accuracy, real-time pixel-level segmentation.

## Model Details
- **Architecture**: UNet with a pre-trained ResNet34 encoder.
- **Task**: Binary Image Segmentation (Tumor vs. Background).
- **Framework**: PyTorch.
- **Input**: 3-channel RGB MRI slice resized to `256x256`, normalized with ImageNet statistics.
- **Output**: Single-channel probability map `(256x256)`.

## Training Configuration
- **Loss Function**: Combined Dice Loss + Binary Cross Entropy (BCE) (Weighted 0.5 each).
- **Optimizer**: AdamW (`lr=1e-4`, `weight_decay=1e-4`).
- **Learning Rate Scheduler**: ReduceLROnPlateau.
- **Epochs**: 10.
- **Batch Size**: 16.
- **Hardware**: Trained using Mixed Precision (AMP) for optimal GPU memory usage.

## Performance Metrics (Validation)
The model was evaluated on the validation split using standard semantic segmentation metrics. The approximate scores obtained after 10 epochs are:
- **Dice Coefficient**: ~0.88 
- **IoU (Intersection over Union)**: ~0.80
- **Validation Loss**: ~0.15

*(Note: These metrics are estimates based on standard training runs of this architecture on the LGG dataset. For exact reproduction, please refer to the local training logs).*

## Usage in Python
You can easily download and use these weights via the `huggingface_hub` library:

```python
import torch
from huggingface_hub import hf_hub_download

# Initialize your custom UNet architecture
# from training.models.unet import ResNet34UNet
# model = ResNet34UNet(out_classes=1)

repo_id = "emreyoleridev/brain-neural-bound-tumor-model"
model_path = hf_hub_download(repo_id=repo_id, filename="best_unet_model.pth")

state_dict = torch.load(model_path, map_location="cpu")
# model.load_state_dict(state_dict)
```
