import torch
import torch.nn as nn

class ZScoreNormalization(nn.Module):
    def __init__(self):
        super(ZScoreNormalization, self).__init__()

    def z_score_normalization(self, x):
        # 在最后一个维度计算均值和标准差
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)
        # 标准化，加入小的常数避免除0
        return (x - mean) / (std + 1e-5)

    def forward(self, x1, x2):
        # 对 x1 和 x2 进行归一化
        x1_norm = self.z_score_normalization(x1)
        x2_norm = self.z_score_normalization(x2)
        
        # 将 x1 和 x2 连接起来
        result = torch.cat((x1_norm, x2_norm), dim=-1)
        return result
