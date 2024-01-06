import torch
from torch.optim import Optimizer
from torch.optim import SGD
from torch.optim import Adagrad
from model import NMF

PARAM_NAMES = {'W', 'H'}


class RMISOBCD(Optimizer):

    def __init__(self, params, target_rows=28, n_components=15, n_nodes=50, dynamic_reg=False, rho=0, beta=1):
        if not 0.0 <= rho:
            raise ValueError("Invalid proximal reg paramter : {}".format(rho))
        if not 0.0 <= beta:
            raise ValueError("Invalid scaling constant : {}".format(beta))
        try:
            named_params = set([group['name'] for group in params])
            assert named_params == PARAM_NAMES, "optimizer must have exactly two parameter groups named 'W' and 'H'"
        except KeyError:
            raise ValueError("optimizer must have exactly two parameter groups named 'W' and 'H'")

        defaults = dict(dynamic_reg=dynamic_reg, rho=rho, beta=beta)
        super().__init__(params, defaults)
        # save indices of dictionary and code for easier access
        self.i_H = 0
        self.i_W = 0
        for i in range(len(self.param_groups)):
            group = self.param_groups[i]
            assert len(group['params']) == 1, "each parameter group should only contain one parameter"
            name = group['name']
            if name == 'W':
                self.i_W = i
            elif name == 'H':
                self.i_H = i
        # store A = \sum_{v \in \V} H_{k^v(n)}H_{k^v(n)}^T
        self.state['A'] = torch.zeros(n_components, n_components)
        # store B = \sum_{v \in \V} H_{k^v(n)}X_v^T
        self.state['B'] = torch.zeros(n_components, target_rows)
        self.state['X'] = None
        self.n_nodes = n_nodes
        self.state['return_times'] = torch.zeros(n_nodes)
        self.state['node'] = 0

    def _update_dict(self, sub_iter=10, stopping_tol=1e-1):
        i = 0
        loss_diff = torch.inf
        loss_old = 1e9
        A = self.state['A']
        B = self.state['B']
        group = self.param_groups[self.i_W]
        W = group['params'][0]
        W_old = W.detach().clone()

        if group['dynamic_reg']:
            returns = self.state['return_times']
            node = self.state['curr_node']
            returns.add_(torch.ones(self.n_nodes))
            returns[node] = 0
            group['rho'] = group['beta']*torch.max(returns)

        step_size = 1/(torch.trace(A) + group['rho'])
        while i < sub_iter and loss_diff > stopping_tol:
            self.zero_grad()
            loss = self.avg_surg_reg(W, W_old, A, B, group['rho'])
            loss.backward()
            grad = W.grad
            W.data.add_(-grad, alpha=step_size)
            W.data.clamp_(min=0)
            gamma = 1/max(1, torch.norm(W).item())
            W.data.mul_(gamma)
            i += 1
            loss_diff = abs(loss.item() - loss_old)
            loss_old = loss.item()

    def _update_code(self, closure, sub_iter=10, stopping_tol=0.1, initialization=False):
        group = self.param_groups[self.i_H]
        loss_diff = torch.inf
        loss_old = 1e9
        i = 0
        W = self.param_groups[self.i_W]['params'][0]
        A = torch.matmul(W.T, W)
        step_size = 1/(torch.trace(A).item())
        assert len(group['params']) == 1
        H = group['params'][0]
        H_old = H.detach().clone()
        while i < sub_iter and loss_diff > stopping_tol:
            self.zero_grad()
            loss = closure()
            loss.backward()
            grad = H.grad
            H.data.add_(-grad, alpha=step_size)
            H.data.clamp_(min=0)
            i += 1
            loss_diff = abs(loss.item() - loss_old)
            loss_old = loss.item()
        #print(i)

        if not initialization:
            with torch.no_grad():
                A_diff = torch.matmul(H, H.T) - torch.matmul(H_old, H_old.T)
                self.state['A'].add_(A_diff, alpha=1/self.n_nodes)
                X = self.state['X']
                B_diff = torch.matmul(H, X.T) - torch.matmul(H_old, X.T)
                self.state['B'].add_(B_diff, alpha=1/self.n_nodes)

    def init_surrogate(self, closure=None):
        if closure is None:
            raise ValueError("Optimizer requires closure argument")
        self._update_code(closure, sub_iter=500, stopping_tol=1e-2, initialization=True)

    def step(self, closure=None):
        if closure is None:
            raise ValueError("Optimizer requires closure argument")
        self._update_code(closure, sub_iter=500, stopping_tol=1e-2)
        self._update_dict(sub_iter=500, stopping_tol=1e-3)

    def set_data_node(self, node, X):
        self.state['X'] = X.clone()
        self.curr_node = node

    @staticmethod
    def avg_surg_reg(W, W_old, A, B, rho):
        AW = torch.matmul(W, torch.matmul(A, W.T))
        WB = torch.matmul(W, B)
        return torch.trace(AW) - 2*torch.trace(WB) + rho/2*torch.norm(W - W_old)**2


class PSGD(SGD):
    def __init__(self, params, lr, **kwargs):
        super().__init__(params, lr, **kwargs)

    def step(self, closure=None):
        super().step()
        for group in self.param_groups:
            for p in group['params']:
                # enforce non-negativity
                p.data.clamp_(min=0)
                if group['name'] == 'W':
                    # project onto unit ball
                    gamma = 1/max(1, torch.norm(p).item())
                    p.data.mul_(gamma)


class PAdagrad(Adagrad):
    def __init__(self, params, lr, **kwargs):
        super().__init__(params, lr, **kwargs)

    def step(self, closure=None):
        super().step()
        for group in self.param_groups:
            for p in group['params']:
                # enforce non-negativity
                p.data.clamp_(min=0)
                if group['name'] == 'W':
                    # project onto unit ball
                    gamma = 1/max(1, torch.norm(p).item())
                    p.data.mul_(gamma)


def test():
    W = torch.randn(2, 1).clamp_(min=0)
    H = torch.randn(1, 3).clamp_(min=0)
    model = NMF(n_components=1, W=W, H=H)
    X = torch.randn(2, 3).clamp_(min=0)
    A = torch.matmul(H, H.T)
    B = torch.matmul(H, X.T)
    groups = [
        {'params': model.params.W, 'name': 'W'},
        {'params': model.params.H, 'name': 'H'}
    ]
    optimizer = RMISOBCD(groups, 2, 1, rho=100)
    optimizer.state['A'] = A
    optimizer.state['B'] = B
    #optimizer.set_data_matrix(X)
    optimizer.set_data_node(0, X)
    loss = torch.nn.MSELoss()
    closure = lambda: loss(X, model.forward()) + 1e-2*torch.norm(model.params.H.data, p=1)
    optimizer.step(closure=closure)

if __name__ == "__main__":
    test()
