import torch
import torch.nn as nn
import numpy as np

class Embedding(nn.Module):
    # embedding_dim:潜在的语义特征维度,用多少个维度来表示一个词的含义(d_model),num_embeddings对应vocab_size
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        w = self.initialize_weight(self.num_embeddings, self.embedding_dim, self.device, self.dtype)
        self.weight = nn.Parameter(w)

    def initialize_weight(self, num_embeddings, embedding_dim, device, dtype):
        mean = 0
        std = 1
        W = torch.empty(self.num_embeddings, self.embedding_dim, device = self.device, dtype = self.dtype)
        nn.init.trunc_normal_(W, mean, std, -3, 3)
        return W
    
    # torch.tensor创建张量,是一个函数,torch.Tensor是一个类构造函数,创建特定类型的张量(float32)
    # 如果给 torch.Tensor() 传入整数，它会认为你是在指定张量的形状，并创建一个该形状的、内容未初始化的空张量。而 torch.tensor() 则会认为你是在用这些整数创建张量的内容。
    def forward(self, token_ids:torch.Tensor) -> torch.Tensor: 
        """
        Args:
            token_ids:形状为(batch_size, sequence_length)

        Returns:
            (batch_size, sequence_length, embedding_dim)
        """
        return self.weight[token_ids]
        