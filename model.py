import torch
import torch.nn as nn
import torch.nn.functional as F


class FlappyNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=input_dim, out_channels=32, kernel_size=5, stride=2)
        #38x38x32
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        #18x18x64
        self.conv_layer3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        #8x8x128
        self.linear_layer1 = nn.Linear(8*8*128, 512)
        self.linear_layer2 = nn.Linear(512, output_dim)
        
    def forward(self, input_data):
        x = F.relu(self.conv_layer1(input_data))
        x = F.relu(self.conv_layer2(x))
        x = F.relu(self.conv_layer3(x))
        #Flatten all dimensions expect the batch
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.linear_layer1(x))
        x = self.linear_layer2(x)
        return x

