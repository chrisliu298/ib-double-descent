import torch.nn as nn
import torch.nn.functional as F


class FCN(nn.Module):
    def __init__(self, layer_config, activation):
        super().__init__()
        # The model config should be a list of integers, where each integer
        # represents the number of neurons in a layer.
        # For example, 784x512x256x128x64x10 represents a 784-512-256-128-64-10 FCN.
        self.layer_config = [int(x) for x in layer_config.split("x")]
        self._layers = []
        for i in range(1, len(self.layer_config) - 1):
            layer = nn.Linear(self.layer_config[i - 1], self.layer_config[i])
            self._layers.append(layer)
            self.add_module(f"layer{i}", layer)
        self.fc = nn.Linear(self.layer_config[-2], self.layer_config[-1])
        self.activation = activation

    def forward(self, x):
        x = x.flatten(1)
        for layer in self._layers:
            x = self.activation(layer(x))
        x = self.fc(x)
        return x
