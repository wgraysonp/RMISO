import torch
from torch.optim import Optimizer


class MCSAG(Optimizer):

    def __init__(self, params, lr, num_nodes=10, rho=0, tau=1, dynamic_step=False, weight_decay=0, delta=1):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= rho:
            raise ValueError("Invalid step parameter: {}".format(rho))
        defaults = dict(lr=lr, rho=rho, tau=tau, dynamic_step=dynamic_step, delta=delta, weight_decay=weight_decay)
        super(MCSAG, self).__init__(params, defaults)
        self.num_nodes = num_nodes
        self.curr_node = 0

        self.grad_dict = {}
        self.param_dict = {}

    def __setstate__(self, state):
        super(MCSAG, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                if p not in self.grad_dict:
                    self.grad_dict[p] = {}

                if p not in self.param_dict:
                    self.param_dict[p] = {}

                if len(state) == 0:
                    state['step'] = 0
                    if self.grad_dict[p]:
                        grad_list = list(self.grad_dict[p].values())
                        state['avg_grad'] = torch.mean(torch.stack(grad_list), dim=0)
                    else:
                        state['avg_grad'] = torch.zeros_like(p.data)

                    if self.param_dict[p]:
                        param_list = list(self.param_dict[p].values())
                        state['avg_param'] = torch.mean(torch.stack(param_list), dim=0)
                    else:
                        state['avg_param'] = torch.zeros_like(p.data)

                    if group['dynamic_step']:
                        state['return_time'] = torch.zeros(self.num_nodes)

                if group['dynamic_step']:
                    state['return_time'].add_(torch.ones(self.num_nodes))
                    state['return_time'][self.curr_node] = 0
                    group['rho'] = torch.max(state['return_time'])

                state['step'] += 1

                self.grad_dict[p][self.curr_node] = grad.detach().clone()
                grad_list = list(self.grad_dict[p].values())
                avg_grad = torch.mean(torch.stack(grad_list), dim=0)

                state['avg_grad'] = avg_grad

                L = 1/group['lr']
                denom = 2*L*(group['tau'] + group['delta']*group['rho'])
                step_size = 1/denom

                p.data.add_(-avg_grad, alpha=step_size)

        return loss

    def set_current_node(self, node_id):
        self.curr_node = node_id

    def init_params(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                if p not in self.grad_dict:
                    self.grad_dict[p] = {}

                if self.curr_node not in self.grad_dict[p]:
                    self.grad_dict[p][self.curr_node] = grad.detach().clone()
