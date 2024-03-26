from torch import tensor
#from graph import Graph
#import data_graph
import matplotlib.pyplot as plt
import random
import numpy as np


class MetropolisHastings:
    def __init__(self, initial_state=0, graph=None, seed=0):
        if graph is None:
            raise ValueError("Must provide graph object")
        d_max = graph.get_max_degree()
        # transition probabilities: self.probs[i][j] = p(i, j)
        self.probs = {}
        # node id of current state
        self.state = initial_state
        for node in graph.nodes:
            prob = {}
            neighbors = [n for n in graph.neighbors(node)]
            num_neighbors = len(neighbors)
            for next_node in graph.nodes:
                if next_node in neighbors:
                    prob[next_node] = 1/d_max
                elif next_node == node:
                    prob[next_node] = 1 - num_neighbors/d_max
                else:
                    prob[next_node] = 0

            self.probs[node] = prob

    def step(self):
        prob = self.probs[self.state]
        choices = list(prob.keys())
        weights = list(prob.values())
        self.state = random.choices(choices, weights=weights).pop()

    def get_state(self):
        return self.state

    def set_seed(self, seed):
        random.seed(seed)


class RandomWalk:
    def __init__(self, initial_state=0, graph=None, seed=0):
        if graph is None:
            raise ValueError("Must provide graph object")
        self.graph = graph
        self.state = initial_state
        random.seed(seed)

    def step(self):
        neighbors = [n for n in self.graph.neighbors(self.state)]
        self.state = random.choices(neighbors).pop()

    def set_seed(self, seed):
        random.seed(seed)


class Uniform:
    def __init__(self, initial_state=0, graph=None, seed=0):
        if graph is None:
            raise ValueError("Must provide graph object")
        self.state = initial_state
        self.graph = graph
        random.seed(seed)

    def step(self):
        choices = list(self.graph.nodes)
        self.state = random.choices(choices).pop()

    def set_seed(self, seed):
        random.seed(seed)


class Sequential:

    def __init__(self, initial_state=0, graph=None):
        if graph is None:
            raise ValueError("Must provide graph object")
        self.state = initial_state
        self.graph = graph

    def step(self):
        N = self.graph.num_nodes
        self.state = (self.state + 1) % N

