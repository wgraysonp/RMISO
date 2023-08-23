import torch
from torch.optim import Optimizer


class RegScheduler(object):

    def __init__(self, optim, name='rho', stepsize=10, gamma=2):
        assert isinstance(optim, Optimizer), "optim must be a torch optimizer object"
        self.optim = optim
        self.name = name
        self.stepsize = stepsize
        self.gamma = gamma
        self.epoch = 0

    def step(self):
        try:
            if self.epoch > 0 and self.epoch % self.stepsize == 0:
                for group in self.optim.param_groups:
                    group[self.name] *= self.gamma
        except KeyError:
            print("Optimizer has no hyper-parameter: {}".format(self.name))
            print("Continuing without adjustment")
            pass
        self.epoch += 1

