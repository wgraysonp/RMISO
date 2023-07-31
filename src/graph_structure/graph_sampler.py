from torch.utils.data.sampler import Sampler
from .sampling_algorithms import Uniform, MetropolisHastings
from .graph import Graph
import pickle
import os


class GraphBatchSampler(Sampler):
    def __init__(self, data_source, algorithm="uniform", load_graph=False, initial_state=0, num_nodes=10, num_edges=10, epoch_length=None):
        self.data_source = data_source

        dir = os.path.join(os.getcwd(), "saved_graphs")
        os.makedirs(dir, exist_ok=True)
        fname = "data_graph.pickle"
        path = os.path.join(dir, fname)

        if load_graph:
            if not os.path.exists(path):
                raise ValueError("No saved graph available")
            print("loading graph -- overriding num_nodes and num_edges")
            self.graph = pickle.load(open(path, "rb"))
        else:
            source_indices = list(range(data_source.__len__()))
            self.graph = Graph(source_indices, num_nodes=num_nodes, num_edges=num_edges)
            if os.path.exists(path):
                os.remove(path)
            pickle.dump(self.graph, open(path, "wb"))

        if epoch_length is not None:
            self.epoch_length = epoch_length
        else:
            self.epoch_length = num_nodes

        if algorithm == "uniform":
            self.sampling_alg = Uniform(initial_state=initial_state, graph=self.graph)
        elif algorithm == "metropolis_hastings":
            self.sampling_alg = MetropolisHastings(initial_state=initial_state, graph=self.graph)

        self.state = self.sampling_alg.state

    def __iter__(self):
        iter_count = 0
        while iter_count < self.epoch_length:
            curr = self.graph.nodes[self.sampling_alg.state]
            batch = curr.data
            iter_count += 1
            yield batch
            # Need to step the sampling algorithm after yielding the bath so that
            # the current sampling state associated with the batch can be
            # accessed in the training loop
            self.sampling_alg.step()
            self.state = self.sampling_alg.state

    def __len__(self):
        return self.graph.num_nodes

    def get_state(self):
        return self.state




