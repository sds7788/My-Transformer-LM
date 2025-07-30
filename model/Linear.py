import torch
import torch.nn as nn
import numpy as np

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """
        构建线性变换模块,初始化

        Args:
        in_features: 输入维度
        out_features: 输出维度
        device: 参数存放设备,默认为None
        dtype: 参数数据类型,默认为None
        """
        # 调用父类的初始化函数
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        # 构造初始化权重w,形状(out_features, int_features)
        w = torch.empty(out_features, in_features, device=device, dtype=dtype)
        
        # 将w设置为nn.parameter,识别为可学习的参数
        self.weight = nn.Parameter(w)

        # 按照要求使用截断正态分布来初始化权重:线性层权重：N(μ=0,σ2=din​+dout​2​)，截断于 [-3, 3]
        mean = 0
        std = np.sqrt(2/ (in_features + out_features))

        nn.init.trunc_normal_(self.weight, mean, std, a = -3.0, b = 3.0)

    def forward(self, x:torch.tensor) -> torch.tensor:
        y = x @self.weight.T
        return y