import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.functional import accuracy
from tqdm import trange


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
        for layer in self._layers:
            x = torch.tanh(layer(x))
        x = self.fc(x)
        return x


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
    x_dec = np.zeros(num_samples)
    y = np.zeros(num_samples)

    for i in range(num_samples):
        x_bin[i, :] = [int(b) for b in list("{:b}".format(i).zfill(input_size))]
        x_dec[i] = i
        y[i] = groups[i % int(num_groups)]

    return x_bin, y.astype(int), x_dec


# parameters
input_size = 10
batch_size = 2**5
layer_config = f"{input_size}x8x6x4x2"
lr = 0.1
num_epochs = 1000
seed = 42

# set device and seed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_everything(seed)

# generate data
x_train, y_train, x_train_int = generate_data(input_size)
x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train)
train_dataset = TensorDataset(x_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# initialize model, optimizer
model = FCN(layer_config).to(device)
print(model)
optimizer = optim.SGD(model.parameters(), lr=lr)

# train
loss_hist = []
acc_hist = []
lr_hist = []
for epoch in range(num_epochs):
    running_loss = 0.0
    running_acc = 0.0
    for i, data in enumerate(train_dataloader):
        x, y = data
        x, y = x.to(device), y.to(device)
        # forward pass
        outputs = model(x)
        # calculate loss and acc
        loss = F.cross_entropy(outputs, y)
        pred = torch.argmax(outputs, dim=1)
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

# plot loss
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(loss_hist)
ax[0].set_title("Loss")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Loss")
ax[1].plot(acc_hist)
ax[1].set_title("Accuracy")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Accuracy")
plt.show()
