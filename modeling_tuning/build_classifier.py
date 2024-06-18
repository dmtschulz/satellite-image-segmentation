import torch.nn as nn
import gc
import torch
import segmentation_models_pytorch as smp

torch.cuda.empty_cache()
gc.collect()

# Define baseline model
class BuildingClassifierBaseline(nn.Module):
    def __init__(self, in_channels):
        super(BuildingClassifierBaseline, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1, padding=0)
        )

    def forward(self, x):
        x = self.model(x).squeeze(1)  # Remove the channel dimension
        return x
    
# Define FastFCN model
class BuildingClassifierUnet(nn.Module):
    def __init__(self, in_channels):
        super(BuildingClassifierUnet, self).__init__()
        self.model = smp.Unet(
            encoder_name='resnet34',  # Use ResNet34 as the encoder
            in_channels=in_channels,  # Input channels
            classes=1,  # Number of classes (binary segmentation)
        )

    def forward(self, x):
        return self.model(x).squeeze(1)  # Remove the channel dimension 