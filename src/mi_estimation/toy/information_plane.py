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


def calculate_layer_mi(layer_out, num_bins, x_id, y):
    num_samples = layer_out.shape[0]
    pdf_x, pdf_y, pdf_t, pdf_xt, pdf_yt = [Counter() for _ in range(5)]
    bins = np.linspace(-1, 1, num_bins + 1)
    indices = np.digitize(layer_out, bins)
    for i in range(num_samples):
        pdf_x[x_id[i]] += 1 / num_samples
        pdf_y[y[i].item()] += 1 / num_samples
        pdf_xt[(x_id[i],) + tuple(indices[i, :])] += 1 / num_samples
        pdf_yt[(y[i].item(),) + tuple(indices[i, :])] += 1 / num_samples
        pdf_t[tuple(indices[i, :])] += 1 / num_samples
    i_xt = sum(
        pdf_xt[i] * np.log2(pdf_xt[i] / (pdf_x[i[0]] * pdf_t[i[1:]])) for i in pdf_xt
    )
    i_yt = sum(
        pdf_yt[i] * np.log2(pdf_yt[i] / (pdf_y[i[0]] * pdf_t[i[1:]])) for i in pdf_yt
    )
    return i_xt, i_yt


def run_mi_estimation(train_dataloader, x_test, y_test, x_test_id):
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
        for _, data in enumerate(train_dataloader):
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
            _, Ts = model(x_test.to(device))
        layer_i_xt_at_epoch = {}
        layer_i_yt_at_epoch = {}
        for layer_idx, t in enumerate(Ts, 1):
            i_xt, i_yt = calculate_layer_mi(
                t.cpu().numpy(), num_bins, x_test_id, y_test
            )
            layer_i_xt_at_epoch[f"l{layer_idx}"] = i_xt
            layer_i_yt_at_epoch[f"l{layer_idx}"] = i_yt
        layer_i_xt_at_epoch["epoch"] = layer_i_yt_at_epoch["epoch"] = epoch
        layer_i_xt_hist.append(layer_i_xt_at_epoch)
        layer_i_yt_hist.append(layer_i_yt_at_epoch)
    df_xt = pd.DataFrame(layer_i_xt_hist)
    df_yt = pd.DataFrame(layer_i_yt_hist)
    return df_xt, df_yt


# parameters
input_size = 12
batch_size = 256
layer_config = f"12x10x7x5x4x3x2"
lr = 0.1
num_epochs = 1000
seed = 0
num_bins = 30

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# generate data
data = sio.loadmat("g1.mat")
x, y = data["F"], data["y"]
x_id = np.arange(x.shape[0])
x = torch.from_numpy(x).float()
y = torch.from_numpy(y).flatten()
indices = torch.randperm(x.shape[0])
train_indices = indices[: int(0.85 * x.shape[0])]
test_indices = indices[int(0.85 * x.shape[0]) :]
x_train, y_train = x[train_indices], y[train_indices]
x_test, y_test = x[test_indices], y[test_indices]
x_train_id, x_test_id = x_id[train_indices], x_id[test_indices]
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# plot layer MI
df_xt_runs = []
df_yt_runs = []
for i in range(5):
    df_xt, df_yt = run_mi_estimation(train_dataloader, x_test, y_test, x_test_id)
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
            norm=colors.LogNorm(vmin=1, vmax=df_xt["epoch"].max()),
        )
    fig.colorbar(a, label="Epoch")
    plt.show()
    plt.close()
