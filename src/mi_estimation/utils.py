from collections import Counter
from math import log2

import matplotlib.pyplot as plt
import torch
from matplotlib import colors


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
        pdf_xt[i] * log2(pdf_xt[i] / (pdf_x[i[0]] * pdf_t[i[1:]])) for i in pdf_xt
    )
    i_yt = sum(
        pdf_yt[i] * log2(pdf_yt[i] / (pdf_y[i[0]] * pdf_t[i[1:]])) for i in pdf_yt
    )
    return i_xt, i_yt


def plot_mi(df_i_xt, df_i_yt, num_cols, timestamp):
    plt.figure(figsize=(8, 6))
    plt.xlabel(r"$I(X; T)$")
    plt.ylabel(r"$I(Y; T)$")
    for i in range(num_cols):
        plt.scatter(
            df_i_xt[f"l{i+1}_i_xt"],
            df_i_yt[f"l{i+1}_i_yt"],
            s=50,
            c=df_i_xt["epoch"],
            cmap="viridis",
            norm=colors.LogNorm(vmin=1, vmax=df_i_xt["epoch"].max()),
        )
    plt.colorbar(label="Epoch")
    plt.savefig(f"information_plane_{timestamp}.pdf", bbox_inches="tight")
    plt.savefig(f"information_plane_{timestamp}.png", bbox_inches="tight", dpi=600)


@torch.no_grad()
def weight_stats(module):
    means = {}
    stds = {}
    norms = {}
    weights = []
    for pn, p in module.named_parameters():
        means[f"weight_mean_{pn}"] = p.data.mean()
        stds[f"weight_std_{pn}"] = p.data.std()
        norms[f"weight_norm_{pn}"] = p.data.norm(p=2)
        weights.append(p.data.flatten())
    means["weight_mean_all"] = torch.cat(weights).mean()
    stds["weight_std_all"] = torch.cat(weights).std()
    norms["weight_norm_all"] = torch.cat(weights).norm(p=2)
    return {**means, **stds, **norms}


@torch.no_grad()
def grad_stats(module):
    means = {}
    stds = {}
    norms = {}
    grads = []
    for pn, p in module.named_parameters():
        if p.grad is not None:
            means[f"grad_mean_{pn}"] = p.grad.data.mean()
            stds[f"grad_std_{pn}"] = p.grad.data.std()
            norms[f"grad_norm_{pn}"] = p.grad.data.norm(p=2)
            grads.append(p.grad.data.flatten())
    means["grad_mean_all"] = torch.cat(grads).mean()
    stds["grad_std_all"] = torch.cat(grads).std()
    norms["grad_norm_all"] = torch.stack(grads).norm(p=2)
    return {**means, **stds, **norms}


def log_now(epoch):
    # Taken from https://github.com/artemyk/ibsgd/tree/iclr2018
    # Only log activity for some epochs.  Mainly this is to make things run faster.
    if epoch < 20:  # Log for all first 20 epochs
        return True
    elif epoch < 100:  # Then for every 5th epoch
        return epoch % 5 == 0
    elif epoch < 2000:  # Then every 10th
        return epoch % 20 == 0
    else:  # Then every 100th
        return epoch % 100 == 0
