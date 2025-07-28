# 实现缩放点积注意力和因果多头自注意力
import torch
import math
import torch.nn as nn
from einops import rearrange, einsum
from cs336_basics.RoPE import RoPE
from cs336_basics.Linear import Linear

def Softmax(x: torch.tensor, dim: int = -1):
    """
    实现softmax函数

    Args:
        x:输入张量
        dim:操作维度,默认为最后一个维度
    """
    # 沿着这个维度找到最大值,减掉,防止e指数溢出inf
    max_val = torch.max(x, dim=dim, keepdim=True).values
    exps = torch.exp(x - max_val)

    # 沿着指定维度对指数结果求和
    sum_exps = torch.sum(exps, dim=dim, keepdim=True)

    # 将指数结果除以其和，得到归一化的概率分布
    return exps / sum_exps

def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor = None) -> torch.Tensor:
    """
    计算缩放点积注意力。

    Args:
        q (torch.Tensor): 查询张量，形状为 (..., seq_len_q, d_k)。
        k (torch.Tensor): 键张量，形状为 (..., seq_len_k, d_k)。
        v (torch.Tensor): 值张量，形状为 (..., seq_len_v, d_v)，其中 seq_len_k == seq_len_v。
        mask (torch.Tensor, optional): 布尔掩码，形状为 (..., seq_len_q, seq_len_k)。
                                       值为 True 的位置允许关注，False 的位置将被屏蔽。默认为 None。

    Returns:
        torch.Tensor: 注意力机制的输出，形状为 (..., seq_len_q, d_v)。
    """
    # 获取键向量的维度d_k,缩放因子
    d_k = q.size(-1)

    # 计算查询(Q)和键(K)的点积，并进行缩放
    attn_scores = einsum(q, k, "... seq_len_q d_k, ... seq_len_k d_k -> ... seq_len_q seq_len_k") / math.sqrt(d_k)

    # 应用掩码
    if mask is not None:
        attn_scores = attn_scores.masked_fill(mask == False, -1e9)

    # 对注意力分数应用 softmax，得到注意力权重,权重矩阵的每一行和为1。
    attention_weights = Softmax(attn_scores, dim=-1)

    # 将注意力权重乘以值(V)向量,即加权
    output = attention_weights @ v

    return output

class CausalMultiHeadSelfAttention(nn.Module):
    """
    因果多头自注意力机制
    """
    def __init__(self, d_model: int, num_heads: int, use_rope: bool=False, max_seq_len: int | None=None,
                theta: float | None=None, token_positions: torch.Tensor | None=None):
        """
        Args:
            d_model (int): 模型的总维度。
            num_heads (int): 注意力头的数量。
            use_rope:是否使用旋转位置编码
            max_seq_len:最大序列长度
            theta:rope使用的θ
            token_positions:位置索引
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_rope = use_rope
        # 初始化rope准备
        self.rope = RoPE(theta, d_model // num_heads, max_seq_len) if use_rope else None
        self.token_positions = token_positions
        self.q_proj = Linear(d_model, d_model) # Q 投影
        self.k_proj = Linear(d_model, d_model) # K 投影
        self.v_proj = Linear(d_model, d_model) # V 投影
        self.o_proj = Linear(d_model, d_model) # 输出线性层

    def forward(self, in_features:torch.Tensor):
        """            
        Args:
            in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run implementation on.

        Returns:
            Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
        """
        seq_len = in_features.shape[-2]
        qkv_proj = torch.cat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight])
        qkv = in_features @ qkv_proj.T
        q, k, v = qkv.chunk(3, -1)

        # 这里使用einops中的rearrange,快速拆分便于理解
        q = rearrange(q, "... seq_len (h d_head) -> ... h seq_len d_head", h=self.num_heads)
        k = rearrange(k, "... seq_len (h d_head) -> ... h seq_len d_head", h=self.num_heads)
        v = rearrange(v, "... seq_len (h d_head) -> ... h seq_len d_head", h=self.num_heads)

        # 应用位置编码
        if self.use_rope:
            q = self.rope(q, self.token_positions)
            k = self.rope(k, self.token_positions)

        # 创建并应用因果掩码,进行缩放点积注意力
        casual_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        casual_mask = casual_mask[None, None, :, :]
        output = scaled_dot_product_attention(q, k, v, ~casual_mask) # ~ 是将掩码反转

        # 再次使用rearrange合并多头
        output = rearrange(
            output, "... h seq_len d_head ->  ... seq_len (h d_head)"
        )

        return self.o_proj(output)

