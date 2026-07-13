from typing import Optional, List, Tuple
import math
import functools
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from omegaconf import ListConfig
from src.models.nn.components import ReversibleInstanceNorm1dInput, ReversibleInstanceNorm1dOutput, \
    TSNormalization, TSInverseNormalization

from src.models.nn.adaptive_softmax import AdaptiveEmbedding, ProjectedAdaptiveLogSoftmax
import src.tasks.metrics as M
from src.tasks.torchmetrics import torchmetric_fns as tm_mine
import src.models.nn.utils as U
import torchmetrics as tm
from src.utils.config import to_list, instantiate
from torchmetrics import MetricCollection
from src.models.sequence.crossattention import CrossAttention
from src.models.sequence.cnn import CNNNET
from src.models.sequence.gatefusion import GatedFusion
from src.models.sequence.bidirectional_cross_attention import BidirectionalCrossAttention
from src.models.sequence.SPECTRELayer import SPECTRELayer
from src.models.sequence.norm import ZScoreNormalization

import os




def get_unique_path(save_path):
    """
    如果文件存在，则自动在文件名后加 _1, _2, ...
    """
    base, ext = os.path.splitext(save_path)
    counter = 1

    new_path = save_path
    while os.path.exists(new_path):
        new_path = f"{base}_{counter}{ext}"
        counter += 1

    return new_path

def plot_position_interaction_map(
        representation,        # (B, L, D) or (L, D)
        batch_idx=0,
        seq_len_show=32,
        figsize=(4, 4),
        cmap="YlGnBu",
        title=None,
        save_path=None,
    ):
        """
        Position × Position interaction heatmap
        """



            # cmap="Reds"
        #     cmap="Blues"
# cmap="coolwarm"
# cmap="hot"
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns


            # ⭐ 关键1：保证字体可嵌入（论文很重要）
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42
    
        if representation.dim() == 3:
            h = representation[batch_idx, :seq_len_show]
        else:
            h = representation[:seq_len_show]
    
        h = h.detach().cpu().numpy()  # (L, D)
    
        # position-position similarity
        sim = h @ h.T                 # (L, L)
        sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-10)
    
        plt.figure(figsize=figsize)
        sns.heatmap(
            sim,
            cmap=cmap,
            square=True,
            xticklabels=False,
            yticklabels=False,
            cbar=True,
        )
    
        plt.xlabel("Position")
        plt.ylabel("Position")
        if title:
            plt.title(title)
    
        plt.tight_layout()
        if save_path:
            # plt.savefig(save_path, bbox_inches="tight")
            save_path = get_unique_path(save_path)   # ⭐ 加这一行
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
    
        return sim




def plot_attention_map(
    attention_weights,          # shape 支持以下几种：
                                # (B, num_heads, L, L)
                                # (B, L, L)          ← 已平均或单头
                                # (L, L)             ← 最简单情况
    layer_name="layer",         # 用于标题
    head_idx=None,              # 如果是多头，可以指定看哪一头；None 则平均所有头
    batch_idx=0,
    seq_len_show=32,
    figsize=(5.2, 4.8),
    cmap="viridis",             # 注意力图常用 viridis / inferno / plasma / RdPu
    title=None,
    xlabel="Key Position",
    ylabel="Query Position",
    save_path=None,
    show=True
):
    """
    可视化 Transformer 某一层的注意力权重热力图
    支持多头 / 单头 / 已平均的情况
    """
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    attn = attention_weights

    # 统一处理成 (L, L)
    if attn.dim() == 4:  # (B, H, L, L)
        if head_idx is not None:
            attn = attn[batch_idx, head_idx, :seq_len_show, :seq_len_show]
            head_info = f" (head {head_idx})"
        else:
            # 平均所有头
            attn = attn[batch_idx].mean(dim=0)[:seq_len_show, :seq_len_show]
            head_info = " (avg heads)"
    elif attn.dim() == 3:  # (B, L, L)
        attn = attn[batch_idx, :seq_len_show, :seq_len_show]
        head_info = ""
    elif attn.dim() == 2:  # (L, L)
        attn = attn[:seq_len_show, :seq_len_show]
        head_info = ""
    else:
        raise ValueError("attention_weights shape not supported")

    attn = attn.detach().cpu().numpy()

    plt.figure(figsize=figsize)
    sns.heatmap(
        attn,
        cmap=cmap,
        square=True,
        xticklabels=False,
        yticklabels=False,
        cbar=True,
        cbar_kws={"shrink": 0.7, "aspect": 30}
    )

    default_title = f"Attention Map – {layer_name}{head_info}"
    plt.title(title if title else default_title, fontsize=13, pad=10)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        if not show:
            plt.close()
    if show:
        plt.show()

    return attn


class BaseTask:
    """ Abstract class that takes care of:
    - loss function
    - arbitrary metrics
    - forward pass
    - (optional) encoder module that interfaces with dataset (inputs) and model
    - (optional) decoder module that interfaces with dataset (targets) and model
    """
    encoder = None
    decoder = None

    def __init__(self, dataset=None, model=None, loss=None, loss_val=None, metrics=None, torchmetrics=None):
        """ This class is allowed to grab attributes directly off a constructed dataset and model object """
        self.dataset = dataset
        self.model = model
        # self.model = model() if callable(model) else deepcopy(model)
        # self.model1 = model() if callable(model) else deepcopy(model)

#         if self.encoder is not None:
#             self.encoder1 = deepcopy(self.encoder)
#             self.encoder2 = deepcopy(self.encoder)

#         if self.decoder is not None:
#             self.decoder1 = deepcopy(self.decoder)
#             self.decoder2 = deepcopy(self.decoder)
        if metrics is None: metrics = []
        
        # print("111111111111111111111111111111111111111111111")
        # print(metrics)
        # print("111111111111111111111111111111111111111111111")
        if 'mcc' not in metrics:
            metrics.append('mcc')
        if 'roc_auc_macro' not in metrics:
            metrics.append('roc_auc_macro')
        if 'f1_binary' not in metrics:
            metrics.append('f1_binary')
            
        self.metric_names = to_list(metrics)

        if torchmetrics is None: torchmetrics = []
        self.torchmetric_names = to_list(torchmetrics)
        self._tracked_torchmetrics = {}

        # The decoder might pass through arguments that the loss needs (e.g. sequence lengths)
        # but might also pass through extraneous arguments (e.g. sampling rate)
        # Wrap loss and metrics so that they accept kwargs and

        # Create loss function
        self.loss = instantiate(M.output_metric_fns, loss, partial=True)
        self.loss = U.discard_kwargs(self.loss)
        if loss_val is not None:
            self.loss_val = instantiate(M.output_metric_fns, loss_val, partial=True)
            self.loss_val = U.discard_kwargs(self.loss_val)
        torchmetrics = MetricCollection(self._init_torchmetrics())
        self.train_torchmetrics = torchmetrics.clone(prefix='train/')
        self.val_torchmetrics = torchmetrics.clone(prefix='val/')
        self.test_torchmetrics = torchmetrics.clone(prefix='test/')
        # 假设你是在某个 Task 类里定义的
        self.cross_attention = CrossAttention(
            in_dim1=128, in_dim2=128, k_dim=128, v_dim=128, num_heads=1
        
        )
        self.cross_attention1 = CrossAttention(
            in_dim1=128, in_dim2=128, k_dim=64, v_dim=64, num_heads=1
        )
        self.cnnnet=CNNNET(input_channel=128).half().cuda()
        
        self.gatefusion = GatedFusion(in_dim1=128, in_dim2=128, d_model=128)
        self.gatefusion_lm = GatedFusion(in_dim1=200, in_dim2=200, d_model=128)
        
        self.bidfusion = BidirectionalCrossAttention(
            dim=128,          # x 的最后一个维度（char 分词特征维度）
            context_dim=128,   # context 的最后一个维度（BPE 分词特征维度）
            dim_head=32,           # 每个注意力头的维度（可调）
            heads=1,               # 注意力头数（可调）
            dropout=0.0,           # 注意力 Dropout
            prenorm=True           # 是否 LayerNorm 在前（推荐 True）
        ).to('cuda')
        self.bidfusion_lm = BidirectionalCrossAttention(
            dim=200,          # x 的最后一个维度（char 分词特征维度）
            context_dim=200,   # context 的最后一个维度（BPE 分词特征维度）
            dim_head=32,           # 每个注意力头的维度（可调）
            heads=1,               # 注意力头数（可调）
            dropout=0.0,           # 注意力 Dropout
            prenorm=True           # 是否 LayerNorm 在前（推荐 True）
        ).to('cuda')
        
        self.dropout = nn.Dropout(p=0.6).to('cuda')  # 0.5 是常用值，可以调
        self.spectre = SPECTRELayer(
            d_model=128,         # 保持与 x/context 的特征维度一致
            n_heads=1,           # 与 BCA 中的 heads 数保持一致
            max_seq_len=1028,     # 根据你的输入序列长度设（例如文本序列）
            low_rank=8,          # 可选，提升频域建模质量
            use_wavelet=True,    # 可选，加入 WRM 改善局部建模
            dropout=0.0          # 与 BidFusion 保持一致
        ).to('cuda')
        
        self.fuse_linear = nn.Linear(256, 128).to('cuda')
        
        self.norm=ZScoreNormalization().to('cuda')

    
        
        
        
    def z_score_normalization(x):
        mean = torch.mean(x, dim=-1, keepdim=True)  # 在最后一个维度计算均值
        std = torch.std(x, dim=-1, keepdim=True)  # 在最后一个维度计算标准差
        return (x - mean) / (std + 1e-5)  # 标准化，加入小的常数避免除0





    def _init_torchmetrics(self):
        """
        Instantiate torchmetrics.
        """
        tracked_torchmetrics = {}

        for name in self.torchmetric_names:
            if name in tm_mine:
                tracked_torchmetrics[name] = tm_mine[name]()
            elif name == 'MCC' or name == 'MatthewsCorrcoef':
                tracked_torchmetrics[name] = tm.MatthewsCorrcoef(num_classes=self.dataset.d_output, average='macro', compute_on_step=False)
            elif name in ['AUROC', 'StatScores', 'Precision', 'Recall', 'F1', 'F1Score']:
                tracked_torchmetrics[name] = getattr(tm, name)(average='macro', num_classes=self.dataset.d_output, compute_on_step=False)
            elif '@' in name:
                k = int(name.split('@')[1])
                mname = name.split('@')[0]
                tracked_torchmetrics[name] = getattr(tm, mname)(average='macro', num_classes=self.dataset.d_output, compute_on_step=False, top_k=k)
            else:
                tracked_torchmetrics[name] = getattr(tm, name)(compute_on_step=False)
        
        return tracked_torchmetrics

    def _reset_torchmetrics(self, prefix=None):
        """
        Reset torchmetrics for a prefix
        associated with a particular dataloader (e.g. train, val, test).

        Generally do this at the start of an epoch.
        """
        all_prefixes = [prefix] if prefix is not None else self._tracked_torchmetrics

        for prefix in all_prefixes:
            if prefix in self._tracked_torchmetrics:
                self._tracked_torchmetrics[prefix].reset()

    def get_torchmetrics(self, prefix):
        """
        Compute torchmetrics for a prefix associated with
        a particular dataloader (e.g. train, val, test).

        Generally do this at the end of an epoch.
        """
        return {name: self._tracked_torchmetrics[prefix][name].compute() for name in self.torchmetric_names}

    def torchmetrics(self, x, y, prefix, loss=None):
        """
        Update torchmetrics with new x, y .
        Prefix corresponds to a particular dataloader (e.g. train, val, test).

        Generally call this every batch.
        """
        if prefix not in self._tracked_torchmetrics:
            self._init_torchmetrics(prefix)
        self._tracked_torchmetrics[prefix](x, y, loss=loss)

        # for name in self.torchmetric_names:
        #     if name.startswith('Accuracy'):
        #         if len(x.shape) > 2:
        #             # Multi-dimensional, multi-class
        #             self._tracked_torchmetrics[prefix][name].update(x.transpose(1, 2), y.squeeze())
        #             continue
        #     self._tracked_torchmetrics[prefix][name].update(x, y)

    def get_torchmetrics(self, prefix):
        return self._tracked_torchmetrics[prefix]

    def metrics(self, x, y, **kwargs):
        """
        Metrics are just functions
        output metrics are a function of output and target
        loss metrics are a function of loss (e.g. perplexity)
        """
        # print(f"Using metrics: {self.metric_names}")
        output_metrics = {
            name: U.discard_kwargs(M.output_metric_fns[name])(x, y, **kwargs)
            for name in self.metric_names if name in M.output_metric_fns
        }
        loss_metrics = {
            name: U.discard_kwargs(M.loss_metric_fns[name])(x, y, self.loss, **kwargs)
            for name in self.metric_names if name in M.loss_metric_fns
        }
        return {**output_metrics, **loss_metrics}

    def forward(self, batch, encoder, model, decoder, _state):
        """Passes a batch through the encoder, backbone, and decoder"""
        # z holds arguments such as sequence length
        # x, y, *z = batch # z holds extra dataloader info such as resolution
        
        
        x1, y1, x2,y2,*z = batch # z holds extra dataloader info such as resolution
        if len(z) == 0:
            z = {}
        else:
            assert len(z) == 1 and isinstance(z[0], dict), "Dataloader must return dictionary of extra arguments"
            z = z[0]

        x1, w1 = encoder(x1, **z) # w can model-specific constructions such as key_padding_mask for transformers or state for RNNs
        x1, state1 = model(x1, **w1, state=_state)
        # self._state = state
        # x1, w1 = decoder(x1, state=state, **z)
        
        
        x2, w2 = encoder(x2, **z) # w can model-specific constructions such as key_padding_mask for transformers or state for RNNs
        x2, state2 = model(x2, **w2, state=_state)
        # self._state = state
        # x2, w2 = decoder(x2, state=state, **z)
        
        x = CrossAttention.forward(x1,x2)
        # state = crossAttention.forward(state1,state2)
        
        x,w=decoder(x,state=state1,**z)
        
        
        
        return x, y, w


class Scalar(nn.Module):
    def __init__(self, c=1):
        super().__init__()
        self.c = c
    def forward(self, x):
        return x * self.c

class LMTask(BaseTask):
    def forward(self, batch, encoder, model, decoder, _state, encoder_aux=None, model_aux=None,bidfusion_lm=None,gatefusion_lm=None,crossatt=None,fuse_linear=None , spectre=None, bidatt=None , attlayer=None , selfatt=None):
        """Passes a batch through the encoder, backbone, and decoder"""
#         # z holds arguments such as sequence length
#         x, y, *z = batch # z holds extra dataloader info such as resolution
#         if len(z) == 0:
#             z = {}
#         else:
#             assert len(z) == 1 and isinstance(z[0], dict), "Dataloader must return dictionary of extra arguments"
#             z = z[0]
#         x, w = encoder(x, **z) # w can model-specific constructions such as key_padding_mask for transformers or state for RNNs
#         x, state = model(x, **w, state=_state)
#         self._state = state
#         x, w = decoder(x, state=state, **z)

#         x = x.logits
#         x = rearrange(x, '... C -> (...) C')
#         y = rearrange(y, '... -> (...)')

#         return x, y, w
    
    
        x1, y, x2,y2,*z = batch # z holds extra dataloader info such as resolution
        if len(z) == 0:
            z = {}
        else:
            assert len(z) == 1 and isinstance(z[0], dict), "Dataloader must return dictionary of extra arguments"
            z = z[0]

        x1, w1 = encoder(x1, **z) # w can model-specific constructions such as key_padding_mask for transformers or state for RNNs
        x1, state1 = model(x1, **w1, state=_state)
        # self._state = state
        # x1, w1 = decoder(x1, state=state, **z)
        
        
        x2, w2 = encoder(x2, **z) # w can model-specific constructions such as key_padding_mask for transformers or state for RNNs
        x2, state2 = model(x2, **w2, state=_state)



        
        
        
        
        x1,w1=decoder(x1,state=state1,**z)
        
        x2,w2=decoder(x2,state=state2,**z)
        


        
        # device = next(self.cross_attention.parameters()).device  # 获取模块设备
        x1 = x1.logits
        x2 = x2.logits
        
        # print("x1.shape =", x1.shape)
        # print("x2.shape =", x2.shape)


        
        # x = self.cross_attention(x1, x2)
        x1 , x2 = bidfusion_lm(x1,x2)
        x1 = gatefusion_lm(x1, x2)

        
        x1 = rearrange(x1, '... C -> (...) C')
        y = rearrange(y, '... -> (...)')
        
        return x1, y, w1


class MultiClass(BaseTask):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.continual_metrics = {}
        for name in self.metric_names:
            if name.endswith('_per_class'):
                for spec_idx, spec in enumerate(self.dataset.species):
                    self.continual_metrics[name + '_' + spec] = M.output_metric_fns[name](spec_idx)

    def metrics(self, x, y, **kwargs):
        output_metrics = {}
        for name in self.metric_names:
            if name in M.output_metric_fns:
                if name.endswith('_per_class'):
                    for spec_idx, spec in enumerate(self.dataset.species):
                        self.continual_metrics[name + '_' + spec] = self.continual_metrics[name + '_' + spec].to(x.device)
                        self.continual_metrics[name + '_' + spec].update(x, y)
                        output_metrics[name + '_' + spec] = self.continual_metrics[name + '_' + spec].compute()
                elif name in ['precision', 'recall']:
                    self.continual_metrics[name] = self.continual_metrics[name].to(x.device)
                    output_metrics[name] = self.continual_metrics[name](x, y)
                else:
                    output_metrics[name] = U.discard_kwargs(M.output_metric_fns[name])(x, y, **kwargs)

        loss_metrics = {
            name: U.discard_kwargs(M.loss_metric_fns[name])(x, y, self.loss, **kwargs)
            for name in self.metric_names if name in M.loss_metric_fns
        }

        return {**output_metrics, **loss_metrics}

    def _reset_torchmetrics(self, prefix=None):
        super()._reset_torchmetrics(prefix)
        for name in self.metric_names:
            if name.endswith('_per_class'):
                for spec_idx, spec in enumerate(self.dataset.species):
                    self.continual_metrics[name + '_' + spec].reset()


class MaskedMultiClass(MultiClass):
   
    def forward(self, batch, encoder, model, decoder, _state ,encoder_aux=None, model_aux=None,bidfusion=None,gatefusion=None,crossatt=None,fuse_linear=None , spectre=None, bidatt=None , attlayer=None , selfatt=None ,selfatt0=None , cnnnet=None ,  win_att=None , win_att0=None , sam=None , codon=None):
        """Passes a batch through the encoder, backbone, and decoder"""

        # z holds arguments such as sequence length
        #x, y, *z = batch # z holds extra dataloader info such as resolution
        #if len(z) == 0:
        #    z = {}
        #else:
          #  assert len(z) == 1 and isinstance(z[0], dict), "Dataloader must return dictionary of extra arguments"
         #   z = z[0]

        #x, w = encoder(x) # w can model-specific constructions such as key_padding_mask for transformers or state for RNNs
        #x, state = model(x)
        #self._state = state
        #x, w = decoder(x, state=state, **z)
        #return x, y, w
        x1, y, x2,y2,*z = batch # z holds extra dataloader info such as resolution
        if len(z) == 0:
            z = {}
        else:
            assert len(z) == 1 and isinstance(z[0], dict), "Dataloader must return dictionary of extra arguments"
            z = z[0]
            


        x1, w1 = encoder(x1, **z) # w can model-specific constructions such as key_padding_mask for transformers or state for RNNs
        x1, state1 = model(x1, **w1, state=_state)
        # self._state = state
        # x1, w1 = decoder(x1, state=state, **z)
        
        
        if encoder_aux is None:
            # print("encoder注入失败")
            encoder_aux = encoder
        if model_aux is None:
            print("模型注入失败")
            print("模型注入失败")
            print("模型注入失败")
            model_aux = model
        
        # x2, w2 = encoder_aux(x2, **z) # w can model-specific constructions such as key_padding_mask for transformers or state for RNNs
        x2, w2 = encoder(x2, **z)
        

        x2, state2 = model_aux(x2, **w2, state=_state)
        # x2, state2 = model(x2, **w2, state=_state)
        # self._state = state
        # x2, w2 = decoder(x2, state=state, **z)
                
        # print("x1.shape =", x1.shape)
        # print("x2.shape =", x2.shape)
        # x1 = self.dropout(x1)
        
        # x1=selfatt(x1)
        # x2=selfatt(x2)

        # x1 , x2 = self.bidfusion(x1,x2)
        # x1 = self.gatefusion(x1, x2)
        # plot_position_interaction_map(
        #     x1, title="Input 1 (x1)",
        #     seq_len_show=20,
        #     save_path=f"/public/ojsys/eye/wuwencan/lihaokai/output/plot/dualhyena/x1.pdf"
        # )

        # plot_position_interaction_map(
        #     x2, title="Input 2 (x2)",
        #     seq_len_show=20,
        #     save_path=f"/public/ojsys/eye/wuwencan/lihaokai/output/plot/dualhyena/x2.pdf"
        # )

        ################################################################################################################################################
        x3 , x4 = bidfusion(x1,x2)
        #################################################################################################################################################
        # x1 = crossatt(x1 , x2)
        # x2 = crossatt(x2 , x1)
        # plot_position_interaction_map(
        #     x3, title="After bidirectional fusion (x3)",
        #     seq_len_show=20,
        #     save_path=f"/public/ojsys/eye/wuwencan/lihaokai/output/plot/dualhyena/x3.pdf"
        # )
        # plot_position_interaction_map(
        #     x4, title="After bidirectional fusion (x4)",
        #     seq_len_show=20,
        #     save_path=f"/public/ojsys/eye/wuwencan/lihaokai/output/plot/dualhyena/x4.pdf"
        # )
        ##########################################################################################################################################################
        x5 = gatefusion(x3, x4)
        # plot_position_interaction_map(
        #     x5, title="After gated fusion (x5)",
        #     seq_len_show=20,
        #     save_path=f"/public/ojsys/eye/wuwencan/lihaokai/output/plot/dualhyena/x5.pdf"
        # )
        ####################################################################################################################################################################################
        x1 = x5
        # x1 = self.cross_attention(x1, x2)
        # x1 = crossatt(x1 , x2)
        # x2 = crossatt(x2 , x1)
        # x1 = self.dropout(x1)
        # x1 = self.dropout(x1)
        # # x1 = self.cross_attention(x2, x1)
        
        # x1 = sam(x1, x2)
        
#         x1=selfatt(x1)
        
#         x1=win_att(x1)
#         x1=win_att0(x1)
        
#         x1=selfatt0(x1)
        
        
        
        

        # fused = self.norm(x1,x2)

        # 然后进行连接
        # x1 = torch.cat((x1_norm, x2_norm), dim=-1).to('cuda')

        
        # fused = torch.cat([x1, x2], dim=-1).to('cuda')

        # x1 = self.fuse_linear(fused)


        # x1 , x2 = bidatt(x1,x2)
        # x1 = attlayer(x1, x2, x12, x22)


        
        
        # x1 , con = self.spectre(x1)
        # x1 , con = spectre(x1)
        # x2 , con = self.spectre(x2)
        
        # x1 = x1.permute(0, 2, 1)  # [batch, channels, seq_len]
        # # X1=self.cnnnet(x1)
        # X1=cnnnet(x1)
        # x1 = x1.permute(0, 2, 1)  
        # x1 , con = self.spectre(x1)
    
        # x1 = self.dropout(x1)
        # x1 = codon(x1)
        

        
        x,w1=decoder(x1,state=state1,**z)
        
        # x,w1=decoder(x2,state=state2,**z)

        
       # x = self.cross_attention(x1, x2)

        
        x = rearrange(x, '... C -> (...) C')
        y = rearrange(y, '... -> (...)')

        
        return x, y, w1

class HG38Task(LMTask):

    def __init__(self, dataset=None, model=None, loss=None, loss_val=None, metrics=None, torchmetrics=None, last_k_ppl=None, per_token_ppl=None):
        """ Extending LMTask to add custom metrics for HG38 task 
        
        last_k_ppl: config for custom ppl, with hparams to pass with it

        per_token_ppl: config for per token ppl calc, with list of k (ppls) to track

        """
        self.dataset = dataset
        self.model = model
        if metrics is None: metrics = []
        self.metric_names = to_list(metrics)
        self.last_k_ppl = last_k_ppl
        self.per_token_ppl = per_token_ppl

        if torchmetrics is None: torchmetrics = []
        self.torchmetric_names = to_list(torchmetrics)
        self._tracked_torchmetrics = {}

        # The decoder might pass through arguments that the loss needs (e.g. sequence lengths)
        # but might also pass through extraneous arguments (e.g. sampling rate)
        # Wrap loss and metrics so that they accept kwargs and

        # Create loss function
        self.loss = instantiate(M.output_metric_fns, loss, partial=True)
        self.loss = U.discard_kwargs(self.loss)
        if loss_val is not None:
            self.loss_val = instantiate(M.output_metric_fns, loss_val, partial=True)
            self.loss_val = U.discard_kwargs(self.loss_val)
        torchmetrics = MetricCollection(self._init_torchmetrics())
        self.train_torchmetrics = torchmetrics.clone(prefix='train/')
        self.val_torchmetrics = torchmetrics.clone(prefix='val/')
        self.test_torchmetrics = torchmetrics.clone(prefix='test/')

        # Create custom metrics for last k ppl
        # last_k_ppl is a list of dicts (configs), so loop thru them
        if self.last_k_ppl is not None:
            self.custom_ppl_dict = {}
            for k in self.last_k_ppl:
                key_name = "last_" + str(k) + "_ppl"
                # create config
                custom_ppl_config = {"_name_": "last_k_ppl", "k": k, "seq_len": self.dataset.max_length}
                k_ppl_fn = instantiate(M.output_metric_fns, custom_ppl_config, partial=True)
                k_ppl_fn = U.discard_kwargs(k_ppl_fn)
                self.custom_ppl_dict[key_name] = k_ppl_fn

        # Create custom metric for per token ppl
        if self.per_token_ppl is not None:
            per_token_ppl_config = {"_name_": "per_token_ppl", "ks": self.per_token_ppl["ks"], "seq_len": self.dataset.max_length}
            per_token_fn = instantiate(M.output_metric_fns, per_token_ppl_config, partial=True)
            per_token_fn = U.discard_kwargs(per_token_fn)
            self.per_token_fn = per_token_fn

    def metrics(self, x, y, **kwargs):
        """
        Need to modify metrics to include custom metrics
        """
        
        output_metrics = {
            name: U.discard_kwargs(M.output_metric_fns[name])(x, y, **kwargs)
            for name in self.metric_names if name in M.output_metric_fns
        }
        loss_metrics = {
            name: U.discard_kwargs(M.loss_metric_fns[name])(x, y, self.loss, **kwargs)
            for name in self.metric_names if name in M.loss_metric_fns
        }

        # loop thru all custom ppls and add them to output_metrics
        if self.last_k_ppl is not None:
            for key_name, k_ppl_fn in self.custom_ppl_dict.items():
                output_metrics[key_name] = k_ppl_fn(x, y, **kwargs)

        # loop thru all custom ppls and add them to output_metrics
        if self.per_token_ppl is not None:
            # returns k ppl values, (averaged over batch)
            per_k_ppl = self.per_token_fn(x, y, **kwargs)  

            # loop over ks to log metric
            for ind, k in enumerate(self.per_token_ppl["ks"]):
                key_name = "ppl_at_{}".format(k)
                k = k-1  # 0 index in the background
                output_metrics[key_name] = per_k_ppl[ind]  # should be in order

        return {**output_metrics, **loss_metrics}


class AdaptiveLMTask(BaseTask):
    def __init__(
        self,
        div_val,
        cutoffs : List[int],
        tie_weights : bool,
        tie_projs : List[bool],
        init_scale=1.0,
        bias_scale=0.0,
        dropemb=0.0,
        dropsoft=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        n_tokens = self.dataset.n_tokens
        d_model = self.model.d_model
        d_output = self.model.d_output

        encoder = AdaptiveEmbedding(
            n_tokens,
            d_model,
            d_model,
            cutoffs=cutoffs,
            div_val=div_val,
            init_scale=init_scale,
            dropout=dropemb,
        )

        if tie_weights:
            assert d_model == d_output
            emb_layers = [i.weight for i in encoder.emb_layers]
        else:
            emb_layers = None

        # Construct decoder/loss
        emb_projs = encoder.emb_projs
        loss = ProjectedAdaptiveLogSoftmax(
            n_tokens, d_output, d_output,
            cutoffs, div_val=div_val,
            tie_projs=tie_projs,
            out_projs=emb_projs,
            out_layers_weights=emb_layers,
            bias_scale=bias_scale,
            dropout=dropsoft,
        )

        self.encoder = encoder
        self.loss = loss


registry = {
    'base': BaseTask,
    'multiclass': MultiClass,
    'lm': LMTask,
    'hg38': HG38Task,
    "masked_multiclass": MaskedMultiClass,
}
