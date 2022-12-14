import contextlib
from collections import Counter
from math import log2, sqrt

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

activations = {
    # None-saturating
    "hardshrink": F.hardshrink,  # (min, max)
    "softshrink": F.softshrink,  # (min, max)
    "tanhshrink": F.tanhshrink,  # (min, max)
    # Singly-saturating (lower)
    "relu": F.relu,  # (0, max)
    "elu": F.elu,  # (-1, max)
    "softplus": F.softplus,  # (0, max)
    # Singly-saturating (upper)
    "logsigmoid": F.logsigmoid,  # (min, 0)
    # Doubly-saturating
    "sigmoid": torch.sigmoid,  # (0, 1)
    "hardsigmoid": F.hardsigmoid,  # (0, 1)
    "tanh": torch.tanh,  # (-1, 1)
    "hardtanh": F.hardtanh,  # (-1, 1)
    "softsign": F.softsign,  # (-1, 1)
    "relu6": F.relu6,  # (0, 6)
}


@contextlib.contextmanager
def temp_seed(seed):
    """Used a a context manager to temporarily set the seed of the random number generator."""
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


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
    if activation in ["hardshrink", "softshrink", "tanhshrink"]:  # none-saturating
        bins = torch.linspace(t.min(), t.max(), num_bins)
    elif activation in ["relu", "softplus"]:  # singly-saturating (lower)
        bins = torch.linspace(0, t.max(), num_bins)
    elif activation == "elu":  # singly-saturating (lower)
        bins = torch.linspace(-1, t.max(), num_bins)
    elif activation == "logsigmoid":  # singly-saturating (upper)
        bins = torch.linspace(t.min(), 0, num_bins)
    elif activation in ["sigmoid", "hardsigmoid"]:  # doubly-saturating
        bins = torch.linspace(0, 1, num_bins)
    elif activation in ["tanh", "hardtanh", "softsign"]:  # doubly-saturating
        bins = torch.linspace(-1, 1, num_bins)
    elif activation == "relu6":  # doubly-saturating
        bins = torch.linspace(0, 6, num_bins)
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


def plot_mi(df_i, title, num_cols):
    """Plot the mutual information for each layer."""
    mpl.rcParams.update({"font.size": 20})
    fig, ax1 = plt.subplots(figsize=(10, 8))
    ax1.set_xlabel(r"$I(X; T)$")
    ax1.set_ylabel(r"$I(T; Y)$")
    ax1.set_title("_".join(title.split("_")[1:]))
    # ax1.set_xlim(0, 12.5)
    # ax1.set_ylim(0, 1.05)
    # ax2.set_xlabel(r"$I(X; T)$")
    # ax2.set_ylabel(r"$I(T; Y)$")
    # ax2.set_title("Test")
    # ax2.set_xlim(0, 12.5)
    # ax2.set_ylim(0, 1.05)
    for i in range(num_cols):
        mappable1 = ax1.scatter(
            df_i[f"l{i+1}_i_xt_tr"],
            df_i[f"l{i+1}_i_ty_tr"],
            s=400,
            c=df_i["epoch"],
            cmap="viridis",
            norm=mpl.colors.LogNorm(vmin=1, vmax=df_i["epoch"].max()),
        )
    # for i in range(num_cols):
    #     mappable2 = ax2.scatter(
    #         df_i[f"l{i+1}_i_xt_te"],
    #         df_i[f"l{i+1}_i_ty_te"],
    #         s=400,
    #         c=df_i["epoch"],
    #         cmap="viridis",
    #         norm=mpl.colors.LogNorm(vmin=1, vmax=df_i["epoch"].max()),
    #     )
    # divider = make_axes_locatable(ax1)
    # cax = divider.append_axes("right", size="5%", pad=0.25)
    # divider = make_axes_locatable(ax2)
    # cax = divider.append_axes("right", size="5%", pad=0.25)
    fig.colorbar(mappable1, label="Epochs")
    # fig.delaxes(fig.axes[2])
    # fig.colorbar(mappable2, label="Epochs", cax=cax)
    fig.tight_layout()
    fig.savefig(title + ".pdf", bbox_inches="tight")
    fig.savefig(title + ".png", bbox_inches="tight", dpi=300)
    plt.close(fig)


@torch.no_grad()
def weight_stats(module):
    """Calculate the mean, standard deviation, and norm of the weights in a model."""
    stats = {}
    means, stds, norms, weights = [], [], [], []
    for pn, p in module.named_parameters():
        mean = p.data.mean()
        std = p.data.std()
        norm = p.data.norm(p=2)
        stats[f"weight_mean_{pn}"] = mean.item()
        stats[f"weight_std_{pn}"] = std.item()
        stats[f"weight_norm_{pn}"] = norm.item()
        means.append(mean)
        stds.append(std)
        norms.append(norm)
        weights.append(p.data.flatten())
    weights = torch.cat(weights)
    stats["weight_mean_all"] = weights.mean().item()
    stats["weight_std_all"] = weights.std().item()
    stats["weight_norm_all"] = weights.norm(p=2).item()
    return stats


@torch.no_grad()
def grad_stats(module):
    """Calculate the mean, standard deviation, and norm of the gradients in a model."""
    stats = {}
    means, stds, norms, grads = [], [], [], []
    for pn, p in module.named_parameters():
        if p.grad is not None:
            mean = p.grad.data.mean()
            std = p.grad.data.std()
            norm = p.grad.data.norm(p=2)
            stats[f"grad_mean_{pn}"] = mean.item()
            stats[f"grad_std_{pn}"] = std.item()
            stats[f"grad_norm_{pn}"] = norm.item()
            means.append(mean)
            stds.append(std)
            norms.append(norm)
            grads.append(p.grad.data.flatten())
    grads = torch.cat(grads)
    stats["grad_mean_all"] = grads.mean().item()
    stats["grad_std_all"] = grads.std().item()
    stats["grad_norm_all"] = grads.norm(p=2).item()
    return stats


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


@temp_seed(42)
def add_label_noise(labels, label_noise, num_classes):
    """Add i.i.d. label noise to a tensor of labels."""
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


def dict_average(dicts):
    """Average a list of dictionaries."""
    return {k: np.mean([d[k] for d in dicts]) for k in dicts[0]}


@temp_seed(42)
def sample_data(x, y, num_samples):
    """Sample a subset of the data."""
    indices = np.random.choice(len(x), num_samples, replace=False)
    return x[indices], y[indices]


def lr_schedule(lr_schedule_type):
    if lr_schedule_type == "constant":
        return lambda t: 1
    elif lr_schedule_type == "inverse":
        return lambda t: 1 / (0.05 * t + 1)
    elif lr_schedule_type == "inverse_sqrt":
        return lambda t: 1 / sqrt(1 + t)


def activation_fn(activation_fn_type):
    return activations[activation_fn_type]
