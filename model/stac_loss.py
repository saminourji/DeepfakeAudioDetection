import torch
import torch.nn as nn
import torch.nn.functional as F

"""
IMPLEMENTS StacLoss FOR TRAINING
"""

class StacLoss(nn.Module):
    """
    StacLoss is a version of contrastive loss. This is the triplet form; we have A (anchor), P (positive), N (negative)

    Positive (similar, e.g. real-real or fake-fake) pairs are labeled 1 -> minimize squared Euclidean distance
    Negative (dissimilar, e.g. real-fake) pairs are labeled 0 -> penalize if distance < margin, maximize distance
    """
    def __init__(self, margin: float = 1.0):
        """ 
        Constructor for StacLoss.

        As presented in the paper, the margin defines minimum acceptable distance between positive and negative pairs
        such that the network must separate real and fake audio embeddings by AT LEAST this margin of 1.0
        """
        super().__init__()
        self.margin = margin

    def forward(self, d_pos: torch.Tensor, d_neg: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the StacLoss function
        
        Input:
            d_pos: Tensor ||f(A) - f(P)||_2 is L2 distance for positive pair
            d_neg: Tensor ||f(A) - f(N)||_2 is L2 distance for negative pair
        Output:
            A scalar loss value averaged over the mini-batch
        """
        return F.relu(d_pos - d_neg + self.margin).mean()