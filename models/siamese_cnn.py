import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import MultiHeadSelfAttention

"""
MAIN SIAMESE CNN ARCHITECTURE
"""
class AudioBranch(nn.Module):
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.residual_conv = nn.Conv2d(64, 128, kernel_size=1)
        self.attn = MultiHeadSelfAttention(128, num_heads=8)
        self.drop2d = nn.Dropout2d(0.5)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 128)

    def forward(self, x):
        """
        Forward pass for the AudioBranch. Applies convolution, attention, dropout, pooling, 
            and a dense (fc) layer and normalizes the output.
        
        Output:
            L2 normalized 128-D vector, which is what StacLoss tries to push/pull together/apart
        """
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        if x2.shape != x3.shape:
                    x2 = self.residual_conv(x2)  # project (B, 64, H/4, W/4) -> (B, 128, H/4, W/4)

        x3 = x3 + x2

        x = self.attn(x3)
        x = self.drop2d(x)
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return nn.functional.normalize(x, p=2, dim=1)

class SiameseCNN(nn.Module):
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.branch = AudioBranch(in_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        """ Method used in inference / testing to produce similarity score for a single pair of inputs """
        z1, z2 = self.branch(x1), self.branch(x2)
        return F.pairwise_distance(z1, z2)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """ Method to produce a single feature embedding, called once for each of anchor, positive, and negative """
        return self.branch(x)

    def predict_real_fake(self, x1: torch.Tensor, x2: torch.Tensor, threshold: float = 1.0):
        """ Method that predicts 0 (fake/dissimilar) or 1 (real/similar) """
        dist = self.forward(x1, x2)
        return (dist <= threshold).long() # 1 = real/similar, 0 = fake/dissimilar