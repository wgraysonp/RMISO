import torch
from torch import nn
from torch.nn import functional as F


class TwoLayer(nn.Module):
    def __init__(self, p=10):
        super(TwoLayer, self).__init__()
        self.layer1 = nn.Linear(p, 100, bias=False)
        self.layer2 = nn.Linear(100, 1)

    def forward(self, x):
        out = F.softplus(self.layer1(x), beta=1)
        out = self.layer2(out)
        out = F.softsign(out)
        return out


class OneLayer(nn.Module):
    def __init__(self, p=10):
        super(OneLayer, self).__init__()
        self.layer1 = nn.Linear(p, 1, bias=False)

    def forward(self, x):
        #out = F.sigmoid(self.layer1(x))
        out = self.layer1(x)
        return out


def test():
    net = OneLayer()
    y = net(torch.randn(54))
    print(y)

#test()
