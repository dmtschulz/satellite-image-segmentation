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
        super().__init__()

        # Encoder
        self.e11 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(64)
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1) 
        self.bn21 = nn.BatchNorm2d(128)
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1) 
        self.bn22 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) 
    
        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1) 
        self.bn31 = nn.BatchNorm2d(256)
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1) 
        self.bn32 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) 
    
        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1) 
        self.bn41 = nn.BatchNorm2d(512)
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1) 
        self.bn42 = nn.BatchNorm2d(512)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn_d11 = nn.BatchNorm2d(256)
        self.d12 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn_d12 = nn.BatchNorm2d(256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn_d21 = nn.BatchNorm2d(128)
        self.d22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn_d22 = nn.BatchNorm2d(128)

        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn_d31 = nn.BatchNorm2d(64)
        self.d32 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn_d32 = nn.BatchNorm2d(64)

        # Output layer
        self.outconv = nn.Conv2d(64, 1, kernel_size=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        # Encoder
        xe11 = self.relu(self.bn11(self.e11(x)))
        xe12 = self.relu(self.bn12(self.e12(xe11)))
        xp1 = self.pool1(xe12)

        xe21 = self.relu(self.bn21(self.e21(xp1)))
        xe22 = self.relu(self.bn22(self.e22(xe21)))
        xp2 = self.pool2(xe22)

        xe31 = self.relu(self.bn31(self.e31(xp2)))
        xe32 = self.relu(self.bn32(self.e32(xe31)))
        xp3 = self.pool3(xe32)

        xe41 = self.relu(self.bn41(self.e41(xp3)))
        xe42 = self.relu(self.bn42(self.e42(xe41)))

        # Decoder
        xu1 = self.upconv1(xe42)
        xu11 = torch.cat([xu1, xe32], dim=1)
        xd11 = self.relu(self.bn_d11(self.d11(xu11)))
        xd12 = self.relu(self.bn_d12(self.d12(xd11)))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe22], dim=1)
        xd21 = self.relu(self.bn_d21(self.d21(xu22)))
        xd22 = self.relu(self.bn_d22(self.d22(xd21)))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe12], dim=1)
        xd31 = self.relu(self.bn_d31(self.d31(xu33)))
        xd32 = self.relu(self.bn_d32(self.d32(xd31)))

        # Output layer
        out = self.outconv(xd32)

        return out    