# import torch
# import torch.nn as nn
# # from flash_attn import flash_attn_qkvpacked_func

# flash_attn_qkvpacked_func = None
# try:
#     # 常见于 v2/v3
#     from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func  # type: ignore
# except Exception:
#     try:
#         # 有些打包版本把接口放在 functional
#         from flash_attn.functional import flash_attn_qkvpacked_func  # type: ignore
#     except Exception:
#         flash_attn_qkvpacked_func = None

# class LocalFlashAttention(nn.Module):
#     def __init__(self, hidden_size, num_heads, window=(64, 64), p_dropout=0.0):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.num_heads = num_heads
#         self.head_dim = hidden_size // num_heads
#         self.window = window
#         self.p_dropout = p_dropout

#     def forward(self, qkv, deterministic=True):
#         # qkv: (B, S, 3, H, D)
#         out = flash_attn_qkvpacked_func(
#             qkv,
#             dropout_p=self.p_dropout,
#             deterministic=deterministic,
#             window_size=self.window
#         )
#         return out.reshape(qkv.size(0), qkv.size(1), self.hidden_size)  # (B, S, hidden_size)


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class LocalAttention(nn.Module):
#     def __init__(self, hidden_size, num_heads, window=(64, 64), dropout=0.0):
#         super().__init__()
#         assert hidden_size % num_heads == 0
#         self.hidden_size = hidden_size
#         self.num_heads = num_heads
#         self.head_dim = hidden_size // num_heads
#         self.scale = self.head_dim ** -0.5
#         self.window = window
#         self.dropout = nn.Dropout(dropout)

#         # QKV projection
#         self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
#         self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
#         self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
#         self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

#     def forward(self, x, mask=None):
#         """
#         x: (B, S, hidden_size)
#         mask: optional (B, S) where 0=pad
#         """
#         B, S, _ = x.shape

#         # QKV projections
#         q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,S,D)
#         k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
#         v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

#         # 计算注意力分数 (B,H,S,S)
#         scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

#         # 构造局部 mask
#         left, right = self.window
#         idx = torch.arange(S, device=x.device)
#         dist = idx[None, :] - idx[:, None]   # (S,S)，dist[i,j] = j - i
#         local_mask = (dist < -left) | (dist > right)   # True 表示不允许 attend

#         scores = scores.masked_fill(local_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

#         # 如果有 padding mask，也加进去
#         if mask is not None:
#             # mask: (B, S)，需要扩展到 (B,1,1,S)
#             scores = scores.masked_fill(mask[:, None, None, :] == 0, float('-inf'))

#         # softmax + dropout
#         attn = F.softmax(scores, dim=-1)
#         attn = self.dropout(attn)

#         # attention output
#         out = torch.matmul(attn, v)  # (B,H,S,D)
#         out = out.transpose(1, 2).contiguous().view(B, S, self.hidden_size)
#         out = self.out_proj(out)
#         return out


import torch
import torch.nn as nn
import torch.nn.functional as F

class SlidingWindowAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, window=(64, 64), dropout=0.0):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.left, self.right = window
        self.dropout = nn.Dropout(dropout)

        # QKV projection
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, mask=None):
        """
        x: (B, S, hidden_size)
        mask: optional (B, S) where 0=pad
        """
        B, S, _ = x.shape

        # QKV projections
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim)  # (B,S,H,D)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim)

        out = torch.zeros_like(q)  # (B,S,H,D)

        # 遍历序列位置，局部计算
        for i in range(S):
            start = max(0, i - self.left)
            end = min(S, i + self.right + 1)

            qi = q[:, i]  # (B,H,D)
            ki = k[:, start:end]  # (B,L,H,D)
            vi = v[:, start:end]  # (B,L,H,D)

            # (B,H,D) × (B,L,H,D) → (B,H,L)
            scores = torch.einsum("bhd,bthd->bht", qi, ki) * self.scale

            if mask is not None:
                # mask: (B,S) -> (B,L)
                mask_slice = mask[:, start:end]
                scores = scores.masked_fill(mask_slice[:, None, :] == 0, float('-inf'))

            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)

            # (B,H,L) × (B,L,H,D) → (B,H,D)
            out[:, i] = torch.einsum("bht,bthd->bhd", attn, vi)

        out = out.reshape(B, S, self.hidden_size)
        out = self.out_proj(out)
        return out
