from torch import tensor
from .graph import Graph
import matplotlib.pyplot as plt
import random
import numpy as np


class MetropolisHastings:
    def __init__(self, initial_state=0, graph=None):
        if graph is None:
            raise ValueError("Must provide graph object")
        self.d_max = graph.get_max_degree()
        # transition probabilities: self.probs[i][j] = p(i, j)
        self.probs = {}
        # node id of current state
        self.state = initial_state
        for idx in graph.nodes.keys():
            prob = {}
            for i in range(graph.num_nodes):
                adjacent_list = graph.nodes[idx].neighbors
                num_neighbors = len(adjacent_list)
                if i in adjacent_list:
                    prob[i] = 1/self.d_max
                elif i == idx:
                    prob[i] = 1 - num_neighbors/self.d_max
                else:
                    prob[i] = 0

            self.probs[idx] = prob

    def step(self):
        prob = self.probs[self.state]
        choices = list(prob.keys())
        weights = list(prob.values())
        self.state = random.choices(choices, weights=weights).pop()

    def get_state(self):
        return self.state


class Uniform:
    def __init__(self, initial_state=0, graph=None):
        if graph is None:
            raise ValueError("Must provide graph object")
        self.state = initial_state
        self.graph = graph

    def step(self):
        N = self.graph.num_nodes
        self.state = random.choices(list(range(N))).pop()


def test_MH():
    indices = list(range(100))
    graph = Graph(indices, num_nodes=10, num_edges=20)
    mh_sampler = MetropolisHastings(initial_state=0, graph=graph)
    states = np.zeros(10)
    states[0] = 1
    for i in range(1, 1000):
        mh_sampler.step()
        states = states * (i/(i+1))
        states[mh_sampler.state] +=1/(i+1)

    plt.bar(list(range(10)), states)
    plt.show()


def test_uniform():
    indices = list(range(100))
    graph = Graph(indices, num_nodes=10, num_edges=20)
    uniform_sampler = Uniform(initial_state=0, graph=graph)
    states = np.zeros(10)
    states[0] = 1
    for i in range(1, 1000):
        uniform_sampler.step()
        states = states *(i/(i+1))
        states[uniform_sampler.state] += 1/(i+1)

    plt.bar(list(range(10)), states)
    plt.show()


if __name__ == "__main__":
    #test_MH()
    test_uniform()
