import math
import torch
from torch.optim import Optimizer


class RMISO(Optimizer):

    def __init__(self, params, batch_num, dynamic_step=False, L=1, rho=1):
        if not 0.0 <= rho:
            raise ValueError("Invalid regularization parameter: {}".format(rho))
        if not 0.0 <= L:
            raise ValueError("Invalid Lipschitz constant: {}".format(L))
        defaults = dict(lr=1/(rho + L), dynamic_step=dynamic_step, rho=rho, L=L)
        super(RMISO, self).__init__(params, defaults)
        self.batch_num = batch_num

    def __setstate__(self, state):
        super(RMISO, self).__setstate__(state)

    def step(self, batch_id=0, closure=None):
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

                # compute the maximum elapsed time since each node was visited
                if group['dynamic_step']:
                    state['return_time'].add_(torch.ones(self.batch_num))
                    state['return_time'][batch_id] = 0
                    group['rho'] = torch.max(state['return_time'])

                avg_grad = state['avg_grad']
                avg_param = state['avg_param']
                state['step'] += 1

                avg_grad.add_(grad - state['old_grads'][batch_id])
                state['avg_grad'] = avg_grad
                state['old_grads'][batch_id] = grad

                param = p.data
                avg_param.add_(param - state['old_param'][batch_id])
                state['avg_param'] = avg_param
                state['old_param'][batch_id] = param

                param.mul_(group['rho']*group['lr'])
                param.add_(avg_param, alpha=group['L']*group['lr'])
                param.add_(avg_grad, alpha=-group['lr'])

                p.data = param

        return loss



