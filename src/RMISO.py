import torch
from torch.optim import Optimizer


class RMISO(Optimizer):

    def __init__(self, params, lr, num_nodes=10, dynamic_step=False, rho=1):
        # lr is 1/L where L is lipshitz constant. Store it this way so that the learning rate scheduler can be used
        if not 0.0 <= rho:
            raise ValueError("Invalid regularization parameter: {}".format(rho))
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, dynamic_step=dynamic_step, rho=rho)
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

                # TODO: Maybe change this to account for initial surrogates?
                # initalize lagged parameters and gradients to store the parameter
                # and gradient at the time of the last visit to each function by
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
                        state['avg_grad'] = p.grad.data.detach().clone()

                    if self.param_dict[p]:
                        param_list = list(self.param_dict[p].values())
                        state['avg_param'] = torch.mean(torch.stack(param_list), dim=0)
                    else:
                        state['avg_param'] = p.data.detach().clone()

                    if group['dynamic_step']:
                        # time since last visit to each node
                        state['return_time'] = torch.zeros(self.num_nodes)

                # compute the maximum elapsed time since each node was visited
                if group['dynamic_step']:
                    state['return_time'].add_(torch.ones(self.num_nodes))
                    state['return_time'][self.curr_node] = 0
                    group['rho'] = torch.max(state['return_time'])

                avg_grad = state['avg_grad']
                avg_param = state['avg_param']
                state['step'] += 1

                pi = 1/self.num_nodes

                if self.curr_node in self.grad_dict[p]:
                    last_grad = self.grad_dict[p][self.curr_node]
                    avg_grad.add_(grad - last_grad, alpha=pi)
                else:
                    avg_grad.add_(grad, alpha=pi)

                self.grad_dict[p][self.curr_node] = grad.detach().clone()
                state['avg_grad'] = avg_grad

                param = p.data

                if self.curr_node in self.param_dict[p]:
                    last_param = self.param_dict[p][self.curr_node]
                    avg_param.add_(param - last_param, alpha=pi)
                else:
                    avg_param.add_(param, alpha=pi)

                self.param_dict[p][self.curr_node] = param.detach().clone()
                state['avg_param'] = avg_param

                L = 1/group['lr']
                step_size = 1/(L + group['rho'])

                p.data.mul_(group['rho']*step_size)
                p.data.add_(avg_param, alpha=L*step_size)
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

                if p not in self.grad_dict:
                    self.grad_dict[p] = {}

                if p not in self.param_dict:
                    self.param_dict[p] = {}

                if self.curr_node not in self.grad_dict[p]:
                    self.grad_dict[p][self.curr_node] = grad.detach().clone()

                if self.curr_node not in self.param_dict[p]:
                    self.param_dict[p][self.curr_node] = p.data.detach().clone()





