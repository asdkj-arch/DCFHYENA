import torch
import torch.nn as nn

class GatedFusion(nn.Module):
    def __init__(self, in_dim1, in_dim2, d_model):
        super().__init__()
        self.proj_x = nn.Linear(in_dim1, d_model).to('cuda')
        self.proj_y = nn.Linear(in_dim2, d_model).to('cuda')

        self.Ws = nn.Parameter(torch.randn(1, d_model)).to('cuda')
        self.Wh = nn.Parameter(torch.randn(1, d_model)).to('cuda')
        
        self.gate_net = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            # nn.ReLU(),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

    def forward(self, x, y):

        # print(f"[GatedFusion] x device: {x.device}, y device: {y.device}")
        # print(f"[GatedFusion] Ws device: {self.Ws.device}, Wh device: {self.Wh.device}")


        if x.shape[-1] != self.proj_x.in_features:
            raise ValueError(f"x.shape[-1] = {x.shape[-1]} doesn't match proj_x.in_features = {self.proj_x.in_features}")
        if y.shape[-1] != self.proj_y.in_features:
            raise ValueError(f"y.shape[-1] = {y.shape[-1]} doesn't match proj_y.in_features = {self.proj_y.in_features}")

        # 保证 Ws/Wh 跟 batch 在同设备
        # device = self.Ws.device  # Linear 层所在设备
        # dtype = self.Ws.dtype
        # # x = x.to(device)
        # # y = y.to(device)
        # # x = x.to(dtype=torch.float16)
        # # y = y.to(dtype=torch.float16)
        # x = x.to(device=device, dtype=dtype)
        # y = y.to(device=device, dtype=dtype)

        x = self.proj_x(x)
        y = self.proj_y(y)
        # F = torch.sigmoid(self.Ws * x + self.Wh * y)
        F = self.gate_net(torch.cat([x, y], dim=-1))
        

        return F * x + (1 - F) * y