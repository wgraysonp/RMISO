import pickle
import os
import networkx as nx
from graph_structure.data_graph import DataGraph
from graph_structure.sampling_algorithms import MetropolisHastings, Uniform

graph_path = os.path.join(os.getcwd(), "saved_graphs")
print_graph_path = os.path.join(graph_path, "print_graphs")
os.makedirs(print_graph_path, exist_ok=True)


def save_graphs_as_print():
    for name in os.listdir(graph_path):
        graph = pickle.load(open(name, 'rb'))
        assert isinstance(graph, DataGraph)
        for node in graph.nodes:
            node['loader'] = None
        if isinstance(graph.sampling_alg, MetropolisHastings):
            alg = "metropolis_hastings"
        elif isinstance(graph.sampling_alg, Uniform):
            alg = "uniform"
        else:
            raise ValueError("invalid algorithm")

        save_str = 'alg:{}-nodes:{}-edges:{}.pickle'.format(alg, graph.num_nodes, graph.num_edges)
        save_path = os.path.join(print_graph_path, save_str)
        if not os.path.exists(save_path):
            pickle.dump(graph, open(save_str, 'wb'))


if __name__ == "__main__":
    save_graphs_as_print()
