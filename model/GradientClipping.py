import torch
from collections.abc import Iterable

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    # Filter parameters with gradients
    parameters_with_grad = [p for p in parameters if p.grad is not None]
    
    if len(parameters_with_grad) == 0:
        return
    
    # Calculate total L2 norm of all gradients
    total_norm = torch.sqrt(sum(torch.sum(p.grad.pow(2)) for p in parameters_with_grad))
    
    # Calculate clipping coefficient
    clip_coef = max_l2_norm / (total_norm + 1e-6)  # Add small value to avoid division by zero
    
    # If total norm exceeds max_norm, scale down all gradients
    if clip_coef < 1.0:
        for p in parameters_with_grad:
            p.grad.mul_(clip_coef)