---
title: Brain Neural Bound Tumor Annotator
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Brain Neural Bound Tumor Annotator

Brain Neural Bound Tumor Annotator is a medical imaging segmentation tool designed to assist in identifying and segmenting brain tumors from MRI scans. It uses a custom trained UNet with a ResNet34 backbone to provide accurate, real-time segmentations overlaid on original MRI scans.

## Features
- **Accurate Segmentation**: Utilizes a deep learning model trained on Brain MRI datasets.
- **Interactive UI**: Built with Streamlit for a fast and responsive user experience.
- **Real-time Inference**: Automatically downloads the latest weights from the Hugging Face Model Hub and performs inference on the fly.
