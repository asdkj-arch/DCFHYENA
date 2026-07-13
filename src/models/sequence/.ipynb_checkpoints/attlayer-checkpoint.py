import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.sequence.BiAttention import BiAttention
from src.models.sequence.bidirectional_cross_attention import BidirectionalCrossAttention


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size, q2c_att=True, c2q_att=True , fused_dim=128):
        super(AttentionLayer, self).__init__()
        self.q2c_att = q2c_att
        self.c2q_att = c2q_att
        # self.bi_attention = BiAttention(hidden_size)
        self.bi_attention = BiAttention(
            hidden_size=128,
            q2c_att=True,
            c2q_att=True,
            logit_func='tri_linear'
        ).to('cuda')
        
        
        self.fuse_linear = nn.Linear(hidden_size * (4 if q2c_att else 3), fused_dim)

    def forward(self, h, u, h_a , u_a ,h_mask=None, u_mask=None):
        """
        h: [B, C, d], u: [B, Q, d]
        """
        # if self.q2c_att or self.c2q_att:
        #     u_a, h_a = self.bi_attention(h, u, h_mask, u_mask)
        # else:
        #     u_mean = u.mean(dim=1).unsqueeze(1).expand_as(h)
        #     u_a = u_mean
        #     h_a = None

        if self.q2c_att:
            out = torch.cat([h, u_a, h * u_a, h * h_a], dim=-1)
        else:
            out = torch.cat([h, u_a, h * u_a], dim=-1)
            
        out = self.fuse_linear(out)  # [B, C, fused_dim]

        return out
