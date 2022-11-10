import torch

# These lr schedules below have theoretically guaranteed convergence.
# However, they need to match the corresponding optimizer.
lr_schedules = {
    "inverse": lambda t: 1 / (t + 1),  # for sgd, step wise
    "inverse_slow": lambda t: 1 / (0.05 * t + 1),  # epoch wise
    "inverse_sqrt": lambda t: 1 / (t + 1) ** 0.5,  # for adam, epoch wise
}


@torch.no_grad()
def weight_norm(module):
    """
    Calculate the norm of the weights (per layer and total) for an nn module.
    """
    norms = {}
    weights = []
    # calculate per-layer weight norm
    for pn, p in module.named_parameters():
        norms[f"weight_norm_{pn}"] = p.data.norm(p=2)
        # here we directly compute the sum of the squares to avoid
        # appending the entire weight vector to the list to save memory
        weights.append(p.data.flatten().pow(2).sum())
    # calculate total weight norm
    # here we only need to sum the list of squared weights and take the sqrt
    norms["weight_norm_all"] = torch.stack(weights).sum().sqrt()
    return norms


@torch.no_grad()
def grad_norm(module):
    """
    Calculate the norm of the gradients (per layer and total) for an nn module.
    """
    norms = {}
    grads = []
    # calculate per-layer gradient norm
    for pn, p in module.named_parameters():
        if p.grad is not None:  # some parameters may not have gradients
            norms[f"grad_norm_{pn}"] = p.grad.data.norm(p=2)
            # here we directly compute the sum of the squares to avoid
            # appending the entire weight vector to the list to save memory
            grads.append(p.grad.data.flatten().pow(2).sum())
    # calculate total gradient norm
    # here we only need to sum the list of squared gradients and take the sqrt
    norms["grad_norm_all"] = torch.stack(grads).sum().sqrt()
    return norms
