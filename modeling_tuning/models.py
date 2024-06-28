import torch
import torch.nn as nn

# Define the neural network model
class BaselineCNN(nn.Module):
    def __init__(self, in_channels):
        super(BaselineCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(128, 1, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.relu1(self.conv1(x))

        x = self.relu2(self.conv2(x))

        x = self.relu3(self.conv3(x))
        
        x = self.conv4(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels):
        super(UNet, self).__init__()

        # Encoder
        self.e1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.d2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        # Output layer
        self.outconv = nn.Conv2d(32, 1, kernel_size=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        # Encoder
        xe1 = self.relu(self.e1(x))
        xp1 = self.pool1(xe1)

        xe2 = self.relu(self.e2(xp1))
        xp2 = self.pool2(xe2)

        xe3 = self.relu(self.e3(xp2))

        # Decoder
        xu1 = self.upconv1(xe3)
        xu11 = torch.cat([xu1, xe2], dim=1)
        xd1 = self.relu(self.d1(xu11))

        xu2 = self.upconv2(xd1)
        xu22 = torch.cat([xu2, xe1], dim=1)
        xd2 = self.relu(self.d2(xu22))

        # Output layer
        out = self.outconv(xd2)

        return out