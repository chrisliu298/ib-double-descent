from collections import Counter
from math import log2

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable


def train_test_split(*tensors, total_size, test_size=0.2):
    """Split a sequence of tensors into train and test partitions."""
    indices = torch.randperm(total_size)
    train_indices = indices[: int(total_size * (1 - test_size))]
    test_indices = indices[int(total_size * (1 - test_size)) :]
    for tensor in tensors:
        yield tensor[train_indices]
        yield tensor[test_indices]


def calculate_layer_mi(x_id, t, y, activation, num_bins=30):
    """Calculate the mutual information given the output of a single layer,
    the id's of x, and the labels y.

    The function first calculates p(x), p(y), p(t). It then calculates p(x, t) and
    p(y, t) by counting the number of times each combination of discretized (x, t)
    or (y, t) occurs in the same bucket (i.e., bin). This is done by the Counter()
    trick: the x_id/y concatenated with the corresponding indices (of the bins) are
    used as a hashable key to tell if a particular combination of (x, y) or (y, t)
    has occurred.

    This is adapted from https://github.com/stevenliuyi/information-bottleneck.
    """
    num_samples = t.shape[0]
    pdf_x, pdf_y, pdf_t, pdf_xt, pdf_ty = [Counter() for _ in range(5)]
    # Decide the bin ranges based on the activation function used
    if activation == "tanh":
        bins = torch.linspace(-1, 1, num_bins)
    elif activation == "relu":
        bins = torch.linspace(0, t.max(), num_bins)
    indices = torch.bucketize(t, bins)
    # Calculate probability distributions by counting
    for i in range(num_samples):
        pdf_x[x_id[i].item()] += 1 / num_samples
        pdf_y[y[i].item()] += 1 / num_samples
        pdf_xt[(x_id[i].item(),) + tuple(indices[i, :].tolist())] += 1 / num_samples
        pdf_ty[(y[i].item(),) + tuple(indices[i, :].tolist())] += 1 / num_samples
        pdf_t[tuple(indices[i, :].tolist())] += 1 / num_samples
    # Calculate mutual information by
    # I(X; T) = sum(p(x, y) * log(p(x, y) / (p(x) * p(y))))
    i_xt = sum(
        pdf_xt[i] * log2(pdf_xt[i] / (pdf_x[i[0]] * pdf_t[i[1:]])) for i in pdf_xt
    )
    i_ty = sum(
        pdf_ty[i] * log2(pdf_ty[i] / (pdf_y[i[0]] * pdf_t[i[1:]])) for i in pdf_ty
    )
    return i_xt, i_ty


def plot_mi(df_i, num_cols, timestamp):
    """Plot the mutual information for each layer."""
    mpl.rcParams.update({"font.size": 20})
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(17, 8))
    ax1.set_xlabel(r"$I(X; T)$")
    ax1.set_ylabel(r"$I(T; Y)$")
    ax1.set_title("Train")
    ax1.set_xlim(0, 12.5)
    ax1.set_ylim(0, 1.05)
    ax2.set_xlabel(r"$I(X; T)$")
    ax2.set_ylabel(r"$I(T; Y)$")
    ax2.set_title("Test")
    ax2.set_xlim(0, 12.5)
    ax2.set_ylim(0, 1.05)
    for i in range(num_cols):
        mappable1 = ax1.scatter(
            df_i[f"l{i+1}_i_xt_tr"],
            df_i[f"l{i+1}_i_ty_tr"],
            s=400,
            c=df_i["epoch"],
            cmap="viridis",
            norm=mpl.colors.LogNorm(vmin=1, vmax=df_i["epoch"].max()),
        )
    for i in range(num_cols):
        mappable2 = ax2.scatter(
            df_i[f"l{i+1}_i_xt_te"],
            df_i[f"l{i+1}_i_ty_te"],
            s=400,
            c=df_i["epoch"],
            cmap="viridis",
            norm=mpl.colors.LogNorm(vmin=1, vmax=df_i["epoch"].max()),
        )
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.25)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.25)
    fig.colorbar(mappable1, label="Epochs", cax=cax)
    fig.delaxes(fig.axes[2])
    fig.colorbar(mappable2, label="Epochs", cax=cax)
    fig.tight_layout()
    fig.savefig(f"information_plane_{timestamp}.pdf", bbox_inches="tight")
    fig.savefig(f"information_plane_{timestamp}.png", bbox_inches="tight", dpi=600)


@torch.no_grad()
def weight_stats(module):
    """Calculate the mean, standard deviation, and norm of the weights in a model."""
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
    """Calculate the mean, standard deviation, and norm of the gradients in a model."""
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
    norms["grad_norm_all"] = torch.cat(grads).norm(p=2)
    return {**means, **stds, **norms}


def log_now(epoch):
    """Decide the if the current epoch should be logged.
    Taken from https://github.com/artemyk/ibsgd/tree/iclr2018
    """
    if epoch < 20:  # Log for all first 20 epochs
        return True
    elif epoch < 100:  # Then for every 5th epoch
        return epoch % 5 == 0
    elif epoch < 2000:  # Then every 10th
        return epoch % 20 == 0
    else:  # Then every 100th
        return epoch % 100 == 0


def standardize(x, mean, std):
    """Standardize a tensor to 0 mean and unit variance."""
    return (x - mean) / std


def add_label_noise(labels, label_noise, num_classes):
    labels = np.array(labels)
    # indices for noisy labels
    mask = np.random.rand(len(labels)) < label_noise
    # generate random labels
    random_labels = np.random.choice(num_classes, mask.sum())
    labels[mask] = random_labels
    # convert back to original labels format
    labels = torch.tensor([int(x) for x in labels]).long()
    return labels


def make_binary(x, y, labels):
    binary_label_indices = torch.isin(y, labels)
    x, y = x[binary_label_indices], y[binary_label_indices]
    y = torch.where(y == labels[0], torch.tensor(0), torch.tensor(1))
    return x, y
