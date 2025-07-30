import torch
import torch.nn as nn
import numpy as numpy

# 这里要学习的是增益g_i,大小为d_model,让模型学习到哪一维度的信息更加重要
class RMSNorm(nn.Module):
    def __init__(self, d_model:int, eps:float=1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        w = self.initialize_weight(self.d_model, self.device, self.dtype)
        self.weight = nn.Parameter(w)

    def initialize_weight(self, d_model, device, dtype):
        W = torch.ones(d_model,device=device,dtype=dtype)
        return W
    
    # 计算均方根函数
    def RMS(self, x:torch.Tensor, d_model, eps):
        rms = torch.sqrt((x.pow(2).mean(dim=-1,keepdim=True))+eps)
        return rms


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """处理一个形状为 (batch_size, sequence_length, d_model) 的输入张量，并返回一个相同形状的张量"""
        init_dtype = x.dtype
        x = x.to(torch.float32)

        rms = self.RMS(x, self.d_model, self.eps)
        result = (x / rms) * self.weight

        return result.to(init_dtype)