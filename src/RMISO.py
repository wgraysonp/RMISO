import math
import torch
from torch.optim import Optimizer


class RMISO(Optimizer):

    def __init__(self, params, lr, batch_num=10, dynamic_step=False, rho=1):
        # lr is 1/L where L is lipshitz constant. Store it this way so that the learning rate scheduler can be used
        if not 0.0 <= rho:
            raise ValueError("Invalid regularization parameter: {}".format(rho))
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, dynamic_step=dynamic_step, rho=rho)
        super(RMISO, self).__init__(params, defaults)
        self.batch_num = batch_num
        self.curr_node = 0

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

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['avg_grad'] = torch.zeros_like(p.data)
                    state['avg_param'] = torch.zeros_like(p.data)

                    # TODO: maybe change this to include surrogate initialization
                    # initalize lagged parameters and gradients to store the parameter
                    # and gradient at the time of the last visit to each function by
                    # sampling algorithm
                    state['old_grads'] = [torch.zeros_like(p.data) for i in range(self.batch_num)]
                    state['old_param'] = [torch.zeros_like(p.data) for i in range(self.batch_num)]
                    if group['dynamic_step']:
                        # time since last visit to each node
                        state['return_time'] = torch.zeros(self.batch_num)

                node_id = self.curr_node

                # compute the maximum elapsed time since each node was visited
                if group['dynamic_step']:
                    state['return_time'].add_(torch.ones(self.batch_num))
                    state['return_time'][node_id] = 0
                    group['rho'] = torch.max(state['return_time'])

                avg_grad = state['avg_grad']
                avg_param = state['avg_param']
                state['step'] += 1

                pi = 1/self.batch_num

                avg_grad.add_(grad - state['old_grads'][node_id], alpha=pi)
                state['avg_grad'] = avg_grad
                state['old_grads'][node_id] = grad

                param = p.data
                avg_param.add_(param - state['old_param'][node_id], alpha=pi)
                state['avg_param'] = avg_param
                state['old_param'][node_id] = param

                L = 1/group['lr']
                lmbda = 1/(L + group['rho'])

                param.mul_(group['rho']*lmbda)
                param.add_(avg_param, alpha=L*lmbda)
                param.add_(-avg_grad, alpha=lmbda)

                p.data = param

        return loss

    def set_current_node(self, node_id):
        self.curr_node = node_id



