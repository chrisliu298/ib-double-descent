from collections import Counter

import torch


def train_test_split(*tensors, total_size, test_size=0.2):
    indices = torch.randperm(total_size)
    train_indices = indices[: int(total_size * (1 - test_size))]
    test_indices = indices[int(total_size * (1 - test_size)) :]
    for tensor in tensors:
        yield tensor[train_indices]
        yield tensor[test_indices]


def calculate_layer_mi(layer_out, num_bins, activation, x_id, y):
    num_samples = layer_out.shape[0]
    pdf_x, pdf_y, pdf_t, pdf_xt, pdf_yt = [Counter() for _ in range(5)]
    if activation == "tanh":
        bins = torch.linspace(-1, 1, num_bins + 1)
    elif activation == "sigmoid":
        bins = torch.linspace(0, 1, num_bins + 1)
    elif activation == "relu":
        bins = torch.linspace(0, layer_out.max(), num_bins + 1)
    indices = torch.bucketize(layer_out, bins)
    for i in range(num_samples):
        pdf_x[x_id[i].item()] += 1 / num_samples
        pdf_y[y[i].item()] += 1 / num_samples
        pdf_xt[(x_id[i].item(),) + tuple(indices[i, :].tolist())] += 1 / num_samples
        pdf_yt[(y[i].item(),) + tuple(indices[i, :].tolist())] += 1 / num_samples
        pdf_t[tuple(indices[i, :].tolist())] += 1 / num_samples
    i_xt = sum(
        pdf_xt[i] * torch.log2(pdf_xt[i] / (pdf_x[i[0]] * pdf_t[i[1:]])) for i in pdf_xt
    )
    i_yt = sum(
        pdf_yt[i] * torch.log2(pdf_yt[i] / (pdf_y[i[0]] * pdf_t[i[1:]])) for i in pdf_yt
    )
    return i_xt, i_yt


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
