import torch
import torch.nn as nn

class SAM_Text(nn.Module):
    def __init__(self, d_model=128, reduction=16):
        super(SAM_Text, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // reduction, bias=False),
            # nn.ReLU(inplace=True),
            nn.GELU(),
            nn.Linear(d_model // reduction, d_model, bias=False),
            nn.Sigmoid()
        )
        self.fc_weight = nn.Sequential(
            nn.Linear(d_model, d_model // reduction, bias=False),
            # nn.ReLU(inplace=True),
            nn.GELU(),
            nn.Linear(d_model // reduction, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x_h, x_l):
        # x_h, x_l: [B, L, D]
        b, l, d = x_h.size()

        # ----- High-level -----
        y_h = x_h.mean(dim=1)           # [B, D]，序列平均池化
        h_weight = self.fc_weight(y_h)  # [B, 1]
        y_h = self.fc(y_h).unsqueeze(1) # [B, 1, D]
        x_fusion_h = x_h * y_h * h_weight.unsqueeze(-1)

        # ----- Low-level -----
        y_l = x_l.mean(dim=1)           # [B, D]
        l_weight = self.fc_weight(y_l)  # [B, 1]
        y_l = self.fc(y_l).unsqueeze(1) # [B, 1, D]
        x_fusion_l = x_l * y_l * l_weight.unsqueeze(-1)

        # 融合
        x_fusion = x_fusion_h + x_fusion_l
        return x_fusion  # [B, L, D]
