import networkx as nx
from networkx.utils import pairwise
import random
import matplotlib.pyplot as plt
if __name__ == "__main__":
    from sampling_algorithms import Uniform, MetropolisHastings, RandomWalk, Sequential
else:
    from .sampling_algorithms import Uniform, MetropolisHastings, RandomWalk, Sequential
from torch.utils.data import DataLoader, Subset

class DataGraph(nx.Graph):

    def __init__(self, data_set, num_nodes=10, num_edges=9, initial_state=1, topo='random', radius=0.3,
                 algorithm='uniform', seed=0, sep_classes=False):
        if num_nodes > len(data_set):
            raise ValueError("Number of nodes is larger than dataset")
        if topo == "random":
            if num_edges < num_nodes - 1:
                raise ValueError("Number of edges must be at least {}".format(num_nodes - 1))
            if num_edges > num_nodes*(num_nodes-1)/2:
                raise ValueError("Number of edges can be no larger than {}".format(num_nodes*(num_nodes-1)/2))
        if algorithm not in ['uniform', 'metropolis_hastings', 'random_walk', 'sequential']:
            raise ValueError("Invalid sampling algorithm")

        super().__init__()
        if sep_classes:
            assert num_nodes % len(data_set.classes) == 0, "number of nodes should be divisible by number of classes :{}".format(len(data_set.classes))

        self.sep_classes = sep_classes
        self.data_set = data_set
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.radius = radius
        self.topo = topo
        self.seed = seed
        self.set_seed(seed)
        self.random_gen = random.Random(4)

        self._create_nodes()
        self._connect_graph()

        if algorithm == 'uniform':
            self.sampling_alg = Uniform(initial_state=initial_state, graph=self, seed=seed)
        elif algorithm == 'metropolis_hastings':
            self.sampling_alg = MetropolisHastings(initial_state=initial_state, graph=self, seed=seed)
        elif algorithm == "random_walk":
            self.sampling_alg = RandomWalk(initial_state=initial_state, graph=self, seed=seed)
        elif algorithm == "sequential":
            self.sampling_alg = Sequential(initial_state=initial_state, graph=self)

    def _create_nodes(self):
        if not self.sep_classes:
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
                pos = tuple(self.random_gen.random() for k in range(2))
                self.add_node(i, data=data_idx, pos=pos, loader=loader)
        else:
            nodes_per_label = int(self.num_nodes/len(self.data_set.classes))
            node_count = 0
            node_list = []
            for label in self.data_set.classes:
                idxs = (self.data_set.targets == label).nonzero()[:, 0].tolist()
                N = len(idxs)
                m = int(N/nodes_per_label)
                for i in range(nodes_per_label):
                    if i == nodes_per_label-1:
                        data_idx = idxs[m*i:]
                    else:
                        data_idx = idxs[m*i:m*(i+1)]
                    batch_size = len(data_idx)
                    data_subset = Subset(self.data_set, data_idx)
                    loader = DataLoader(data_subset, batch_size=batch_size, shuffle=False, num_workers=0)
                    pos = tuple(self.random_gen.random() for k in range(2))
                    node_list.append((node_count, {"data": data_idx, "pos": pos, "loader": loader}))
                    #self.add_node(node_count, data=data_idx, pos=pos, loader=loader)
                    node_count += 1
            random.shuffle(node_list)
            self.add_nodes_from(node_list)

    def _connect_graph(self):
        if self.topo == "random":
            self._connect_random()
        elif self.topo == "cycle":
            self._connect_cycle()
            self.num_edges = len(self.edges)
        elif self.topo == "geometric":
            self._connect_geometric(self.radius)
        elif self.topo == 'lonely':
            self._connect_lonely()
        elif self.topo == 'complete':
            self._connect_complete()
        else:
            raise ValueError("Graph topology not supported")

        assert nx.is_connected(self), "Graph is not connected"

    def _connect_cycle(self):
        self.add_edges_from(nx.cycle_graph(self.nodes).edges)

    def _connect_geometric(self, r):
        self.add_edges_from(nx.geometric_edges(self, r))

    def _connect_lonely(self):
        self.add_edges_from(nx.complete_graph(self.nodes).edges)
        for i in range(1, len(self.nodes)-1):
            self.remove_edge(0, i)

    def _connect_complete(self):
        self.add_edges_from(nx.complete_graph(self.nodes).edges)

    def _connect_random(self):

        S, T = list(self.nodes), []

        node_s = self.random_gen.sample(S, 1).pop()
        S.remove(node_s)
        T.append(node_s)

        while S:
            node_s, node_t = self.random_gen.sample(S, 1).pop(), self.random_gen.sample(T, 1).pop()
            self.add_edge(node_s, node_t)
            S.remove(node_s)
            T.append(node_s)

        while self.number_of_edges() < self.num_edges:
            node_1, node_2 = tuple(self.random_gen.sample(list(self.nodes), 2))
            if self.has_edge(node_1, node_2):
                continue
            else:
                self.add_edge(node_1, node_2)

    def set_seed(self, seed):
        random.seed(seed)

    def get_max_degree(self):
        return max(self.degree(node) for node in self.nodes)

    def sample(self):
        state = self.sampling_alg.state
        self.sampling_alg.step()
        return state


def test():
    data_set = list(range(10))
    G = DataGraph(data_set, num_nodes=10, num_edges=11, topo='cycle')
    pos = nx.spring_layout(G)
    nx.draw(G, pos)
    plt.show()
    node = 1
    #print(len(G.nodes[node]['loader']))
    print(len(G.edges))
    for node in G.nodes:
        print([n for n in G.neighbors(node)])


def plot_graph(nodes, edges=100, radius=0.3, topo="random"):
    data_set = list(range(nodes))
    G = DataGraph(data_set, num_nodes=nodes, num_edges=edges, radius=radius, topo=topo)
    if topo == "geometric":
        pos = {node: G.nodes[node]['pos'] for node in G.nodes}
    else:
        pos = nx.spring_layout(G)
    fig, ax = plt.subplots()
    return fig, nx.draw(G, pos, with_labels=True, ax=ax)


if __name__=="__main__":
    test()
