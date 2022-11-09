import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, layer_config, activation):
        super().__init__()
        # The model config should be a list of integers, where each integer
        # represents the number of filters in a layer.
        # For example, 3x64x128x256x512x10 represents a 3-64-128-256-512-10 CNN.
        self.layer_config = [int(x) for x in layer_config.split("x")]
        self._layers = []
        for i in range(1, len(self.layer_config) - 1):
            layer = nn.Conv2d(
                self.layer_config[i - 1], self.layer_config[i], kernel_size=3, padding=1
            )
            self._layers.append(layer)
            self.add_module(f"layer{i}", layer)
        self.fc = nn.Linear(self.layer_config[-2], self.layer_config[-1])
        self.activation = activation

    def forward(self, x):
        for layer in self._layers[:-1]:
            x = self.activation(layer(x))
            x = F.max_pool2d(x, 2)
        x = self.activation(self._layers[-1](x))
        x = F.adaptive_max_pool2d(x, 1)
        x = x.flatten(1)
        x = self.fc(x)
        return x
