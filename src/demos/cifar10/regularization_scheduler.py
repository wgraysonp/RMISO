import torch
from torch.optim import Optimizer


class RegScheduler(object):

    def __init__(self, optim, name='rho', stepsize=10, gamma=2, stop=100,  verbose=False):
        assert isinstance(optim, Optimizer), "optim must be a torch optimizer object"
        self.optim = optim
        self.name = name
        self.stepsize = stepsize
        self.gamma = gamma
        self.epoch = 0
        self.verbose = verbose
        self.stop = stop

    def step(self):
        if self.epoch < self.stop:
            try:
                if self.epoch > 0 and self.epoch % self.stepsize == 0:
                    printed = False
                    for group in self.optim.param_groups:
                        group[self.name] *= self.gamma
                        if self.verbose and not printed:
                            printed = True
                            print("increasing parameter {} to {}".format(self.name, group[self.name]))
            except KeyError:
                if self.verbose:
                    print("Optimizer has no hyper-parameter: {}".format(self.name))
                    print("Continuing without adjustment")
                pass
        self.epoch += 1

