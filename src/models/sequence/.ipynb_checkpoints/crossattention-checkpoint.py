import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, in_dim1, in_dim2, k_dim, v_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim
        
        self.proj_q1 = nn.Linear(in_dim1, k_dim * num_heads, bias=False).to('cuda')
        self.proj_k2 = nn.Linear(in_dim2, k_dim * num_heads, bias=False).to('cuda')
        self.proj_v2 = nn.Linear(in_dim2, v_dim * num_heads, bias=False).to('cuda')
        self.proj_o = nn.Linear(v_dim * num_heads, in_dim1).to('cuda')
        

        
    def forward(self, x1, x2, mask=None):
        batch_size, seq_len1, _ = x1.size()
        seq_len2 = x2.size(1)
        
        x1 = x1.to(dtype=torch.float16, device='cuda')  # 确保是 float16 且在 GPU 上
        
        # q1: (batch_size, num_heads, seq_len1, k_dim)
        q1 = self.proj_q1(x1).view(batch_size, seq_len1, self.num_heads, self.k_dim).permute(0, 2, 1, 3)
        # k2: (batch_size, num_heads, k_dim, seq_len2)
        k2 = self.proj_k2(x2).view(batch_size, seq_len2, self.num_heads, self.k_dim).permute(0, 2, 3, 1)
        # v2: (batch_size, num_heads, seq_len2, v_dim)
        v2 = self.proj_v2(x2).view(batch_size, seq_len2, self.num_heads, self.v_dim).permute(0, 2, 1, 3)
        
        # attention: (batch_size, num_heads, seq_len1, seq_len2)
        attention = torch.matmul(q1, k2) / (self.k_dim ** 0.5)

        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(attention, dim=-1)

        # output: (batch_size, num_heads, seq_len1, v_dim)
        output = torch.matmul(attention, v2).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len1, -1)
        # final projection
        output = self.proj_o(output)
        
        return output
