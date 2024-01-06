import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter


class NMF(nn.Module):

    def __init__(self, W, H, n_components=15):
        super().__init__()

        assert isinstance(W, Tensor)
        assert torch.all(W >= 0.), "Tensor W should be non-negative."
        assert isinstance(H, Tensor)
        assert torch.all(H >= 0.), "Tensor H should be non-negative"

        self.params = nn.ParameterDict({
            'W': Parameter(W, requires_grad=True),
            'H': Parameter(H, requires_grad=True)
        })

        self.n_components = n_components

    def set_code_matrix(self, H):
        assert isinstance(H, Tensor)
        assert torch.all(H >= 0), "Tensor H should be non-negative"
        H = Parameter(H, requires_grad=True)
        with torch.no_grad():
            self.params['H'].copy_(H)

    def get_code_matrix(self):
        return self.params['H'].detach().clone()

    def forward(self):
        W = self.params['W']
        H = self.params['H']
        return torch.matmul(W, H)




def test():
    W = torch.randn(2, 1).clamp_(min=0)
    H = torch.randn(1, 3).clamp_(min=0)
    model = NMF(n_components=1, W=W, H=H)
    X = torch.randn(2, 3)
    loss = nn.MSELoss()
    y = model.forward()
    groups = [
        {'params': model.params.W, 'name': 'W'},
        {'params': model.params.H, 'name': 'H'}
    ]
    optimizer = torch.optim.SGD(groups, lr=1e-3)
    c = loss(X, y)
    c.backward()
    i_W = 0
    i_H = 0
    for i in range(len(optimizer.param_groups)):
        if optimizer.param_groups[i]['name'] == 'W':
            i_W = i
        elif optimizer.param_groups[i]['name'] == 'H':
            i_H = i

    output_rows = optimizer.param_groups[i_H]['params'][0].size()[0]


if __name__=="__main__":
    test()