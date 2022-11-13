from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import colors
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.functional import accuracy

# %matplotlib inline
# %config InlineBackend.figure_format="retina"


class FCN(nn.Module):
    def __init__(self, layer_config):
        super().__init__()
        layer_config = [int(x) for x in layer_config.split("x")]
        self._layers = []
        for i in range(1, len(layer_config) - 1):
            layer = nn.Linear(layer_config[i - 1], layer_config[i])
            self._layers.append(layer)
            self.add_module(f"layer{i}", layer)
        self.fc = nn.Linear(layer_config[-2], layer_config[-1])

    def forward(self, x):
        Ts = []
        for layer in self._layers:
            x = torch.tanh(layer(x))
            Ts.append(x.clone().detach())
        x = self.fc(x)
        Ts.append(x.clone().detach())
        return x, Ts


def generate_data(input_size):
    """
    This data set is created as follows:

    X = binary(num) where num in [0, 2**input_size)
    Y = rand_group[X mod sqrt(2**input_size)]

    If input_size = 10, num_samples = 1024, num_groups = 32, binary_group_size = 16.

    The random group assignment for the labels is done by taking the i-th index of
    a permuted 0-1 vector, where i is the results of the modulo operation.
    """
    num_samples = 2**input_size
    num_groups = np.sqrt(num_samples)
    binary_group_size = int(num_groups / 2)
    groups = np.append(np.zeros(binary_group_size), np.ones(binary_group_size))
    np.random.shuffle(groups)

    x_bin = np.zeros((num_samples, input_size))
    x_idx = np.zeros(num_samples)
    y = np.zeros(num_samples)

    for i in range(num_samples):
        x_bin[i, :] = [int(b) for b in list("{:b}".format(i).zfill(input_size))]
        x_idx[i] = i
        y[i] = groups[i % int(num_groups)]

    return x_bin, y.astype(int), x_idx.astype(int)


def calculate_layer_mi(layer_out, num_bins, x_train_idx, y_train):
    num_samples = layer_out.shape[0]
    pdf_x, pdf_y, pdf_t, pdf_xt, pdf_yt = [Counter() for _ in range(5)]
    i_xt = 0
    i_yt = 0
    bins = np.linspace(-1, 1, num_bins + 1)
    indices = np.digitize(layer_out, bins)

    for i in range(num_samples):
        pdf_x[x_train_idx[i]] += 1 / num_samples
        pdf_y[y_train[i].item()] += 1 / num_samples
        pdf_xt[(x_train_idx[i],) + tuple(indices[i, :])] += 1 / num_samples
        pdf_yt[(y_train[i].item(),) + tuple(indices[i, :])] += 1 / num_samples
        pdf_t[tuple(indices[i, :])] += 1 / num_samples

    for i in pdf_xt:
        p_xt = pdf_xt[i]
        p_x = pdf_x[i[0]]
        p_t = pdf_t[i[1:]]
        i_xt += p_xt * np.log2(p_xt / (p_x * p_t))

    for i in pdf_yt:
        p_yt = pdf_yt[i]
        p_y = pdf_y[i[0]]
        p_t = pdf_t[i[1:]]
        i_yt += p_yt * np.log2(p_yt / (p_y * p_t))

    return i_xt, i_yt


# parameters
input_size = 12
batch_size = 2**input_size
layer_config = f"{input_size}x8x6x4x2"
lr = 0.1
num_epochs = 5000
seed = 0
num_bins = 30

# set device and seed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# seed_everything(seed)

# generate data
x_train, y_train, x_train_idx = generate_data(input_size)
# data = sio.loadmat("var_u.mat")
# x_train, y_train = data["F"], data["y"]
# x_train_idx = np.arange(x_train.shape[0])
x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train)
train_dataset = TensorDataset(x_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


def run_mi_estimation(x_train, y_train):
    # initialize model, optimizer
    model = FCN(layer_config).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_hist = []
    acc_hist = []
    layer_i_xt_hist = []
    layer_i_yt_hist = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_acc = 0.0
        for i, data in enumerate(train_dataloader):
            x, y = data
            x, y = x.to(device), y.to(device)
            # forward pass
            outputs, _ = model(x)
            # calculate loss and acc
            loss = F.cross_entropy(outputs, y)
            pred = outputs.argmax(dim=1)
            acc = accuracy(pred, y)
            # backward pass and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_acc += acc.item()
        avg_loss = running_loss / len(train_dataloader)
        loss_hist.append(avg_loss)
        avg_acc = running_acc / len(train_dataloader)
        acc_hist.append(avg_acc)
        if epoch % (num_epochs // 10) == 0:
            print(
                f"Epoch {epoch:>4}: loss={avg_loss:>.4f}, acc={avg_acc:>.4f} lr={optimizer.param_groups[0]['lr']:.4f}"
            )
        with torch.no_grad():
            _, Ts = model(x_train.to(device))
        layer_i_xt_at_epoch = {}
        layer_i_yt_at_epoch = {}
        for layer_idx, t in enumerate(Ts, 1):
            i_xt, i_yt = calculate_layer_mi(
                t.cpu().numpy(), num_bins, x_train_idx, y_train
            )
            layer_i_xt_at_epoch[f"l{layer_idx}"] = i_xt
            layer_i_yt_at_epoch[f"l{layer_idx}"] = i_yt
        layer_i_xt_at_epoch["epoch"] = layer_i_yt_at_epoch["epoch"] = epoch
        layer_i_xt_hist.append(layer_i_xt_at_epoch)
        layer_i_yt_hist.append(layer_i_yt_at_epoch)
    df_xt = pd.DataFrame(layer_i_xt_hist)
    df_yt = pd.DataFrame(layer_i_yt_hist)
    return df_xt, df_yt


# plot layer MI
df_xt_runs = []
df_yt_runs = []
for i in range(5):
    df_xt, df_yt = run_mi_estimation(x_train, y_train)
    df_xt_runs.append(df_xt)
    df_yt_runs.append(df_yt)
    avg_df_xt = df_xt.copy()
    avg_df_xt.loc[:] = 0
    avg_df_yt = df_yt.copy()
    avg_df_yt.loc[:] = 0
    for df_xt, df_yt in zip(df_xt_runs, df_yt_runs):
        avg_df_xt += df_xt
        avg_df_yt += df_yt
    df_xt = avg_df_xt / len(df_xt_runs)
    df_yt = avg_df_yt / len(df_yt_runs)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel(r"$I(X; T)$")
    ax.set_ylabel(r"$I(Y; T)$")
    num_layers = layer_config.count("x")
    for i in range(num_layers):
        a = ax.scatter(
            df_xt[f"l{i+1}"],
            df_yt[f"l{i+1}"],
            s=50,
            c=df_xt["epoch"],
            cmap="plasma",
            # norm=colors.LogNorm(),
        )
    fig.colorbar(a, label="Epoch")
    plt.show()
    plt.close()
