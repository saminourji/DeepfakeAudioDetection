"""
SELF-ATTENTION MODULE
- use multihead attention
"""

import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        return attn_output + x