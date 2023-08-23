import networkx as nx
from networkx.utils import pairwise
import random
import matplotlib.pyplot as plt
if __name__ == "__main__":
    from sampling_algorithms import Uniform, MetropolisHastings
else:
    from .sampling_algorithms import Uniform, MetropolisHastings
import torch
from torch.utils.data import DataLoader, Subset


class DataGraph(nx.Graph):

    def __init__(self, data_set, num_nodes=10, num_edges=9, initial_state=0, topo='random', algorithm='uniform'):
        if num_nodes > len(data_set):
            raise ValueError("Number of nodes is larger than dataset")
        if num_edges < num_nodes - 1:
            raise ValueError("Number of edges must be at least {}".format(num_nodes - 1))
        if num_edges > num_nodes*(num_nodes-1)/2:
            raise ValueError("Number of edges can be no larger than {}".format(num_nodes*(num_nodes-1)/2))
        if algorithm not in ['uniform', 'metropolis_hastings']:
            raise ValueError("Invalid sampling algorithm")

        super().__init__()

        self.data_set = data_set
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.topo = topo

        self._create_nodes()
        self._connect_graph()

        if algorithm == 'uniform':
            self.sampling_alg = Uniform(initial_state=initial_state, graph=self)
        elif algorithm == 'metropolis_hastings':
            self.sampling_alg = MetropolisHastings(initial_state=initial_state, graph=self)

    def _create_nodes(self):
        N = len(self.data_set)
        idxs = list(range(N))
        m = int(N/self.num_nodes)
        for i in range(self.num_nodes):
            if i == self.num_nodes-1:
                data_idx = idxs[m*i:]
            else:
                data_idx = idxs[m*i:m*(i+1)]
            batch_size = len(data_idx)
            data_subset = Subset(self.data_set, data_idx)
            loader = DataLoader(data_subset, batch_size=batch_size, shuffle=False, num_workers=0)
            self.add_node(i, data=data_idx, loader=loader)

    def _connect_graph(self):
        if self.topo == "random":
            self._connect_random()
        elif self.topo == "cycle":
            self._connect_cycle()
            self.num_edges = len(self.edges)
        else:
            raise ValueError("Graph topology not supported")

        assert nx.is_connected(self), "Graph is not connected"

    def _connect_cycle(self):
        self.add_edges_from(pairwise(self.nodes, cyclic=True))

    def _connect_random(self):
        # set random seed for reproducible graphs
        random.seed(a=4)

        S, T = list(self.nodes), []

        node_s = random.sample(S, 1).pop()
        S.remove(node_s)
        T.append(node_s)

        while S:
            node_s, node_t = random.sample(S, 1).pop(), random.sample(T, 1).pop()
            self.add_edge(node_s, node_t)
            S.remove(node_s)
            T.append(node_s)

        while self.number_of_edges() < self.num_edges:
            node_1, node_2 = tuple(random.sample(list(self.nodes), 2))
            if self.has_edge(node_1, node_2):
                continue
            else:
                self.add_edge(node_1, node_2)

    def get_max_degree(self):
        return max(self.degree(node) for node in self.nodes)

    def sample(self):
        state = self.sampling_alg.state
        self.sampling_alg.step()
        return state


def test():
    data_set = list(range(500))
    G = DataGraph(data_set, num_nodes=7, num_edges=11, topo='cycle')
    pos = nx.spring_layout(G)
    nx.draw(G, pos)
    plt.show()
    node = 1
    #print(len(G.nodes[node]['loader']))
    print(len(G.edges))
    for node in G.nodes:
        print([n for n in G.neighbors(node)])


if __name__=="__main__":
    test()
