import torch.nn as nn
import gc
import torch

torch.cuda.empty_cache()
gc.collect()

class BuildingClassifier(nn.Module):
    def __init__(self, in_channels):
        super(BuildingClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 1, kernel_size=1, padding=0)
        )

    def forward(self, x):
        x = self.model(x).squeeze(1)  # Remove the channel dimension
        return x