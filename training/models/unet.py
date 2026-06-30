import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet34_Weights

class DoubleConv(nn.Module):
    """
    Standard UNet block: Conv2d -> BatchNorm -> ReLU -> Conv2d -> BatchNorm -> ReLU
    Used in the decoder to refine features after upsampling.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class ResNet34UNet(nn.Module):
    """
    U-Net architecture utilizing a pretrained ResNet34 as the encoder.
    The decoder is built manually using transposed convolutions and skip connections.
    """
    def __init__(self, out_classes: int = 1):
        super().__init__()
        
        # ---------------------------------------------------------
        # ENCODER: Pretrained ResNet34
        # ---------------------------------------------------------
        # We load a ResNet34 trained on ImageNet to leverage transfer learning.
        resnet = torchvision.models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        
        # Extract the layers to create skip connections at different resolutions
        # Input shape: [B, 3, 256, 256]
        
        # Skip 1: Stem output before maxpool [B, 64, 128, 128]
        self.encoder_stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )
        self.maxpool = resnet.maxpool # Reduces resolution to [B, 64, 64, 64]
        
        self.encoder_layer1 = resnet.layer1 # Skip 2: [B, 64, 64, 64]
        self.encoder_layer2 = resnet.layer2 # Skip 3: [B, 128, 32, 32]
        self.encoder_layer3 = resnet.layer3 # Skip 4: [B, 256, 16, 16]
        self.bottleneck = resnet.layer4     # Bottom: [B, 512, 8, 8]

        # ---------------------------------------------------------
        # DECODER
        # ---------------------------------------------------------
        # Up1: Upsample Bottom (512) -> Concat with Skip 4 (256) = 768 channels
        self.up1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(512 + 256, 256)
        
        # Up2: Upsample (256) -> Concat with Skip 3 (128) = 384 channels
        self.up2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256 + 128, 128)
        
        # Up3: Upsample (128) -> Concat with Skip 2 (64) = 192 channels
        self.up3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(128 + 64, 64)
        
        # Up4: Upsample (64) -> Concat with Skip 1 (64) = 128 channels
        self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(64 + 64, 64)
        
        # Final Upsampling to reach the original image resolution (256x256)
        self.up_final = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec_final = DoubleConv(32, 32)
        
        # OUTPUT LAYER: 1x1 Convolution to map features to the desired number of classes
        self.out_conv = nn.Conv2d(32, out_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder Pass & Saving Skip Connections
        skip1 = self.encoder_stem(x)         # [B, 64, 128, 128]
        x_pool = self.maxpool(skip1)         # [B, 64, 64, 64]
        
        skip2 = self.encoder_layer1(x_pool)  # [B, 64, 64, 64]
        skip3 = self.encoder_layer2(skip2)   # [B, 128, 32, 32]
        skip4 = self.encoder_layer3(skip3)   # [B, 256, 16, 16]
        bottom = self.bottleneck(skip4)      # [B, 512, 8, 8]
        
        # Decoder Pass (Upsample -> Concat -> DoubleConv)
        up1 = self.up1(bottom)
        dec1 = self.dec1(torch.cat([up1, skip4], dim=1))
        
        up2 = self.up2(dec1)
        dec2 = self.dec2(torch.cat([up2, skip3], dim=1))
        
        up3 = self.up3(dec2)
        dec3 = self.dec3(torch.cat([up3, skip2], dim=1))
        
        up4 = self.up4(dec3)
        dec4 = self.dec4(torch.cat([up4, skip1], dim=1))
        
        up_final = self.up_final(dec4)
        dec_final = self.dec_final(up_final)
        
        # Final output (No Sigmoid here; we will apply it in the loss function for better numerical stability)
        out = self.out_conv(dec_final)       # [B, 1, 256, 256]
        
        return out

if __name__ == "__main__":
    # ---------------------------------------------------------
    # ARCHITECTURE SANITY CHECK
    # ---------------------------------------------------------
    print("[INFO] Testing ResNet34-UNet architecture...")
    
    # Create a dummy batch identical to our DataLoader output [Batch, Channels, Height, Width]
    dummy_input = torch.randn(4, 3, 256, 256)
    
    # Initialize the model
    model = ResNet34UNet(out_classes=1)
    
    # Pass the dummy tensor through the model
    output = model(dummy_input)
    
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {output.shape}")
    
    # Ensure the output shape exactly matches the mask shape [4, 1, 256, 256]
    assert output.shape == (4, 1, 256, 256), "Output shape mismatch!"
    print("[SUCCESS] Model forward pass is valid and ready for training.")