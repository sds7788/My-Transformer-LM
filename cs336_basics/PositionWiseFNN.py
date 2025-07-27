import torch
import torch.nn as nn
from .Linear import Linear

class positionwise_feedforward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        """
        args:
            d_model:模型维度
            d_ff:中间隐藏层维度

        将 SiLU/Swish 和 GLU 放在一起，我们得到了 SwiGLU，我们将用它来构建我们的前馈网络：
        FFN(x)=SwiGLU(x,W1​,W2​,W3​)=W2​(SiLU(W1​x)⊙W3​x),
        其中 x∈Rdmodel​, W1​,W3​∈Rdff​×dmodel​, W2​∈Rdmodel​×dff​，并且通常，dff​=38​dmodel​。
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def Silu(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.w2(self.Silu(self.w1(x)) * self.w3(x))
        return output