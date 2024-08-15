import torch.nn as nn


class Shallow(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(100, 100)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(100, 100)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.sigmoid(self.output(x))
        return x
    