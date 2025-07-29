from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # 获取学习率
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                # 从状态中获取迭代次数 t，如果不存在则初始化为 0
                t = state.setdefault("t", 0)
                
                grad = p.grad.data # 获取损失相对于 p 的梯度
                
                # 就地更新权重张量：p.data = p.data + alpha * grad
                # 这里 alpha 是 -lr / sqrt(t + 1)
                p.data.add_(grad, alpha=-lr / math.sqrt(t + 1))
                
                state["t"] = t + 1 # 增加迭代次数
        return loss
    
weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
opt = SGD([weights], lr=0.01)

for t in range(100):
    opt.zero_grad() # 重置所有可学习参数的梯度
    loss = (weights**2).mean() # 计算一个标量损失值
    print(f"Step {t}: Loss = {loss.cpu().item():.4f}")
    loss.backward() # 运行反向传播，计算梯度
    opt.step() # 运行优化器步骤，更新权重