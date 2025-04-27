import torch
import torch.nn as nn

"""
MULTIHEAD SELF-ATTENTION MODULE
"""

class MultiHeadSelfAttention(nn.Module):
    """
    Multiheaded Attention for 2-D audio feature maps.

    Input:  shape = (B, C, T, F)
    Output: shape = (B, C, T, F) - same shape (residual skip inside).

    Clarifications on shape:
        B - batch_size, e.g. number of "utterances" processed together
        C - channel_dim, e.g. number of feature maps
        T - time_axis, e.g. number of frames or windows in the spectrogram
        F - freq_axis, e.g. number of mel / linear-spectrogram bins per frame
    """
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.ln = nn.LayerNorm(channels)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        Apply multihead self-attention to a 2-D feature map.

        Input:
            x: Tensor of shape (B, C, T, F), 
                where B=batch, C=channels, T=time frames, F=frequency bins.

        Output:
            Tensor of the same shape with context-enhanced features.
        """
        B, C, T, F = x.shape
        seq = x.reshape(B, C, T * F).permute(0, 2, 1) # (B, L, C) where L = T*F
        attn_out, _ = self.mha(seq, seq, seq) # self‑attention
        out = self.gamma * attn_out + seq # residual
        out = self.ln(out)
        out = out.permute(0, 2, 1).reshape(B, C, T, F)
        return out