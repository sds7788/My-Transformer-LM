import torch
import torch.nn as nn

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Args:
            theta:RoPE的θ值
            d_k:查询和键向量的维度 q和k
            max_seq_len:要输入序列的最大长度
            device:储存缓冲区的设备
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        self.precompute_cos_sin()

    def precompute_cos_sin(self):
        # 计算每个维度对的频率 theta_k = theta^(-2k / d_k),形状: (d_k / 2)
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.d_k, 2).float() / self.d_k))
        
        # 创建位置索引 t = 0, 1, ..., max_seq_len-1,形状: (max_seq_len)
        t = torch.arange(self.max_seq_len, device=self.device, dtype=inv_freq.dtype)

        # 计算每个位置和每个维度对的角度 m * theta_k
        freqs = torch.einsum("i,j->ij", t, inv_freq)

        # 计算cos和sin值
        # 使用 register_buffer 将这些张量注册为模块的缓冲区。persistent=False 意味着这些缓冲区不会被包含在模块的state_dict中
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, x:torch.Tensor, token_positions:torch.Tensor) -> torch.Tensor:
        """
        处理一个形状为 (..., seq_len, d_k) 的输入张量，并返回一个相同形状的张量
        注意，你应该容忍 x 具有任意数量的批处理维度。你应该假设词元位置是一个形状为 (seq_len) 的张量，
        指定了 x 沿序列维度的词元位置。你应该使用词元位置来切片你的（可能是预计算的）cos 和 sin 张量沿序列维度。
        tokens_positions形状是(batch_size, seq_len)
        """
        # 1. 将输入 x 的最后一个维度变形，以分离相邻的对
        # x_reshaped: (..., seq_len, d_k/2, 2)
        x_reshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # 2. 提取成对的特征 x_even 和 x_odd
        # 形状都是 (..., seq_len, d_k/2)
        x_even, x_odd = x_reshaped.unbind(-1)

        # 3. 从缓存中获取对应位置的 cos 和 sin 值
        # 形状: (seq_len, d_k/2) -> (1, seq_len, d_k/2)
        # 为了处理token_positions为None的情况
        seq_len = x.shape[-2]
        if token_positions is None:
            # 创建一个从 0 到 seq_len-1 的位置张量
            token_positions = torch.arange(seq_len, device=x.device)

        cos = self.cos_cached[token_positions].unsqueeze(0)
        sin = self.sin_cached[token_positions].unsqueeze(0)

        # 4. 应用正确的旋转公式
        # x_rot_even = x_even * cos - x_odd * sin
        # x_rot_odd = x_even * sin + x_odd * cos
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos

        # 5. 将旋转后的对重新堆叠起来
        # rotated_reshaped: (..., seq_len, d_k/2, 2)
        rotated_reshaped = torch.stack([x_rot_even, x_rot_odd], dim=-1)

        # 6. 将形状恢复为原始输入的形状
        # rotated_x: (..., seq_len, d_k)
        rotated_x = rotated_reshaped.flatten(start_dim=-2)

        return rotated_x.type_as(x)
