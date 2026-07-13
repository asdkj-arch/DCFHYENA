import torch
import torch.nn as nn
import torch.nn.functional as F

# class BiAttention(nn.Module):
#     def __init__(self, hidden_size, q2c_att=True, c2q_att=True, logit_func='tri_linear'):
#         super(BiAttention, self).__init__()
#         self.q2c_att = q2c_att
#         self.c2q_att = c2q_att
#         self.logit_func = logit_func
#         self.hidden_size = hidden_size

#         if logit_func == 'tri_linear':
#             self.linear = nn.Linear(hidden_size * 3, 1)
#         else:
#             raise NotImplementedError(f"Unsupported logit_func: {logit_func}")

#     def get_logits(self, h, u, mask=None):
#         """
#         h: [B, C, d], u: [B, Q, d]
#         输出 logits: [B, C, Q]
#         """
#         B, C, d = h.shape
#         Q = u.shape[1]

#         # 计算三线性拼接向量 [B, C, Q, 3d]
#         # 利用广播生成 [B, C, Q, d]
#         h_exp = h.unsqueeze(2)  # [B, C, 1, d]
#         u_exp = u.unsqueeze(1)  # [B, 1, Q, d]
#         h_repeat = h_exp.expand(-1, C, Q, -1)
#         u_repeat = u_exp.expand(-1, C, Q, -1)
#         h_u_mul = h_repeat * u_repeat

#         concat = torch.cat([h_repeat, u_repeat, h_u_mul], dim=-1)  # [B, C, Q, 3d]
#         logits = self.linear(concat).squeeze(-1)  # [B, C, Q]

#         if mask is not None:
#             logits = logits.masked_fill(~mask, float('-inf'))
#         return logits

#     def softsel(self, values, logits):
#         """
#         values: [B, Q, d], logits: [B, C, Q] or [B, Q]
#         return: [B, C, d] or [B, d]
#         """
#         a = F.softmax(logits, dim=-1)
#         if values.dim() == 3:  # [B, Q, d]
#             return torch.bmm(a, values)  # [B, C, d] or [B, d]
#         else:
#             raise ValueError("Unexpected values shape in softsel")

#     def forward(self, h, u, h_mask=None, u_mask=None):
#         """
#         h: [B, C, d], u: [B, Q, d]
#         """
#         B, C, d = h.shape
#         Q = u.shape[1]

#         if h_mask is not None and u_mask is not None:
#             h_mask_aug = h_mask.unsqueeze(2).expand(-1, C, Q)
#             u_mask_aug = u_mask.unsqueeze(1).expand(-1, C, Q)
#             hu_mask = h_mask_aug & u_mask_aug
#         else:
#             hu_mask = None

#         # 注意力打分
#         logits = self.get_logits(h, u, mask=hu_mask)  # [B, C, Q]

#         # Query-to-Context attention
#         u_a = self.softsel(u, logits) if self.q2c_att else None  # [B, C, d]

#         # Context-to-Query attention（最大值）
#         h_logits = logits.max(dim=-1).values if self.c2q_att else None  # [B, C]
#         h_a = self.softsel(h, h_logits) if self.c2q_att else None       # [B, d]
#         if h_a is not None:
#             h_a = h_a.unsqueeze(1).expand(-1, C, -1)  # [B, C, d]

#         return u_a, h_a




class BiAttention(nn.Module):
    def __init__(self, hidden_size, q2c_att=True, c2q_att=True, logit_func='tri_linear'):
        super(BiAttention, self).__init__()
        self.q2c_att = q2c_att
        self.c2q_att = c2q_att
        self.logit_func = logit_func
        self.hidden_size = hidden_size

        if logit_func == 'tri_linear':
            self.linear = nn.Linear(hidden_size * 3, 1)
        else:
            raise NotImplementedError(f"Unsupported logit_func: {logit_func}")

    def get_logits(self, h_aug, u_aug, mask=None):
        h_u = torch.cat([h_aug, u_aug, h_aug * u_aug], dim=-1)  # [B, C, Q, 3d]
        logits = self.linear(h_u).squeeze(-1)  # [B, C, Q]
        if mask is not None:
            logits = logits.masked_fill(~mask, float('-inf'))
        return logits

    def softsel(self, values, logits):
        a = F.softmax(logits, dim=-1)
        return torch.sum(a.unsqueeze(-1) * values, dim=-2)

    def forward(self, h, u, h_mask=None, u_mask=None):
        """
        h: [B, C, d], u: [B, Q, d]
        h_mask: [B, C], u_mask: [B, Q]
        """
        B, C, d = h.shape
        Q = u.size(1)

        h_aug = h.unsqueeze(2).expand(-1, C, Q, -1)
        u_aug = u.unsqueeze(1).expand(-1, C, Q, -1)

        if h_mask is not None and u_mask is not None:
            h_mask_aug = h_mask.unsqueeze(2).expand(-1, C, Q)
            u_mask_aug = u_mask.unsqueeze(1).expand(-1, C, Q)
            hu_mask = h_mask_aug & u_mask_aug
        else:
            hu_mask = None

        logits = self.get_logits(h_aug, u_aug, mask=hu_mask)  # [B, C, Q]

        # Query-to-Context attention
        u_a = self.softsel(u_aug, logits) if self.q2c_att else None  # [B, C, d]

        # Context-to-Query attention
        h_logits = logits.max(dim=-1).values if self.c2q_att else None
        h_a = self.softsel(h, h_logits) if self.c2q_att else None
        if h_a is not None:
            h_a = h_a.unsqueeze(1).expand(-1, C, -1)

        return u_a, h_a
