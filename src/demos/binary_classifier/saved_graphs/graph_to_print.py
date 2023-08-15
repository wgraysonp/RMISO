import pickle
import os
import networkx as nx
import sys
sys.path.insert(0, "../")
from graph_structure.data_graph import DataGraph
from graph_structure.sampling_algorithms import MetropolisHastings, Uniform
import matplotlib.pyplot as plt
import argparse


print_graph_path = os.path.join(os.getcwd(), "graph_pics")
os.makedirs(print_graph_path, exist_ok=True)


def save_graphs_as_print(f_name):
    graph = pickle.load(open(f_name, 'rb'))
    assert isinstance(graph, DataGraph)
    if isinstance(graph.sampling_alg, MetropolisHastings):
        alg = "metropolis_hastings"
    elif isinstance(graph.sampling_alg, Uniform):
        alg = "uniform"
    else:
        raise ValueError("invalid algorithm")

    save_name = "alg:{}-nodes:{}-edges:{}.pdf".format(alg, graph.num_nodes, graph.num_edges)

    save_str = os.path.join(print_graph_path, save_name)

    fig = plt.figure()
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, ax=fig.add_subplot())
    fig.savefig(save_str)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Print an image of a graph")
    parser.add_argument("--file_name", type=str, help="file name of graph to print")
    args = parser.parse_args()

    save_graphs_as_print(args.file_name)
