import torch
from torch.optim import Optimizer


class RMISO(Optimizer):

    def __init__(self, params, lr, num_nodes=10, dynamic_step=False, rho=1, delta=1e-5):
        # lr is 1/L where L is lipshitz constant. Store it this way so that the learning rate scheduler can be used
        if not 0.0 <= rho:
            raise ValueError("Invalid regularization parameter: {}".format(rho))
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= delta:
            raise ValueError("Invalid multiplier: {}".format(delta))
        if not dynamic_step:
            delta = 1
        defaults = dict(lr=lr, dynamic_step=dynamic_step, rho=rho, delta=delta)
        super(RMISO, self).__init__(params, defaults)
        self.num_nodes = num_nodes
        self.curr_node = 0

        # dictionary to store past gradients and parameters
        self.grad_dict = {}
        self.param_dict = {}

    def __setstate__(self, state):
        super(RMISO, self).__setstate__(state)

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

                # initialize lagged dict to store the parameter
                # and gradient at the time of the last visit to each node by
                # sampling algorithm
                if p not in self.grad_dict:
                    self.grad_dict[p] = {}

                if p not in self.param_dict:
                    self.param_dict[p] = {}

                # State initialization
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
                        # time since last visit to each node
                        state['return_time'] = torch.zeros(self.num_nodes)

                # compute the maximum elapsed time since each node was visited
                if group['dynamic_step']:
                    state['return_time'].add_(torch.ones(self.num_nodes))
                    state['return_time'][self.curr_node] = 0
                    group['rho'] = torch.max(state['return_time'])

                state['step'] += 1

                self.grad_dict[p][self.curr_node] = grad.detach().clone()

                grad_list = list(self.grad_dict[p].values())
                avg_grad = torch.mean(torch.stack(grad_list), dim=0)
                state['avg_grad'] = avg_grad

                param = p.data

                self.param_dict[p][self.curr_node] = param.detach().clone()
                param_list = list(self.param_dict[p].values())
                avg_param = torch.mean(torch.stack(param_list), dim=0)
                state['avg_param'] = avg_param

                L = 1/group['lr']
                step_size = 1/(L + group['delta']*group['rho'])
                alpha = group['delta']*group['rho']*step_size

                avg_param_reg = avg_param.clone()

                avg_param_reg.mul_(1 - alpha)
                avg_param_reg.add_(param, alpha=alpha)

                p.data = avg_param_reg.add(-avg_grad, alpha=step_size)

        return loss

    def set_current_node(self, node_id):
        self.curr_node = node_id

    def init_params(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                if p not in self.grad_dict:
                    self.grad_dict[p] = {}

                if p not in self.param_dict:
                    self.param_dict[p] = {}

                if self.curr_node not in self.grad_dict[p]:
                    self.grad_dict[p][self.curr_node] = grad.detach().clone()

                if self.curr_node not in self.param_dict[p]:
                    self.param_dict[p][self.curr_node] = p.data.detach().clone()





