from nmf import *
import networkx as nx
import numpy as np
from sklearn.datasets import fetch_openml
import numpy.linalg as LA
import matplotlib.pyplot as plt
import random
from sampling_algorithms import Uniform, MetropolisHastings, RandomWalk
from tqdm import tqdm
import argparse
import os
import time
import pickle

GRAPH_TOPOS = ['geometric']
DATASETS = ['mnist']
FULL_BATCH_ALGS = ['mu', 'psgd']
MINI_BATCH_ALGS = ['rmiso', 'minibatch_cd', 'psgd', 'adagrad']


def get_parser():
    parser = argparse.ArgumentParser(description="Distributed Non-Negative Matrix Factorization")
    parser.add_argument('--sampling_algorithm', default='uniform', type=str, help='algorithm to sample from graph',
                        choices=['uniform', 'metropolis_hastings', 'random_walk'])
    parser.add_argument('--graph_size', default=50, type=int, help='number of vertices in the graph')
    parser.add_argument('--graph_topo', default='geometric', type=str, help='graph topology', choices=['geometric'])
    parser.add_argument('--radius', default=0.3, type=float, help='connection radius for geometric graph')
    parser.add_argument('--dataset', default='mnist', type=str, help='data set to train dictionary on')
    parser.add_argument('--alpha', default=0, type=float, help='L1 regularization parameter for sparse coding')
    parser.add_argument('--n_components', default=15, type=int, help='number of dictionary components (columns in the matrix)')
    parser.add_argument('--n_examples', default=100, type=int, help='number of training data pointsw')
    parser.add_argument('--optim', default='rmiso', type=str, help='optimizer',
                        choices=['rmiso', 'minibatch_cd', 'mu', 'psgd', 'adagrad'])
    parser.add_argument('--lr', default=1e-3, type=float, help='adagrad and psgd learning rate')
    parser.add_argument('--eta', default=0.0, type=float, help='adagrad and psgd step decay parameter')
    parser.add_argument('--rho', default=0, type=float, help='rmiso prox reg parameter')
    parser.add_argument('--dynamic_reg', default=False, type=bool, help='use dynamic regularization for rmiso')
    parser.add_argument('--beta', default=1, type=float, help='scaling factor for rmiso dynamic prox reg')
    parser.add_argument('--iterations', default=100, type=int, help='number of iterations to run')
    parser.add_argument('--full_batch', action='store_true', help="turn on full-batch flag for full-batch algs like MU")

    return parser


def build_dataset(dataset, num_examples=10):
    assert dataset in DATASETS, "invalid dataset"
    random_gen = random.Random(4)
    if dataset == "mnist":
        X = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser="auto")[0]
        X = X/255.0
        n = len(X)
        X = X.reshape(n, 28, 28)
        idx = random_gen.sample(range(n), num_examples)
        dataset = X[idx]
        dims = (28, 28)

    return dataset, dims


def build_full_batch(data):
    n = len(data)
    X = data[0]
    for i in range(1, n):
        X = np.hstack((X, data[i]))
    return X


# !!! only run this when there is one data matrix per node for now
def build_graph(data, num_nodes=10, radius=0.3, topo="geometric", n_components=15):
    assert topo in GRAPH_TOPOS, "invalid graph topology"
    assert len(data) >= num_nodes, "not enough nodes"

    random_gen = random.Random(4)
    graph = nx.Graph()
    N = len(data)
    idxs = list(range(N))
    m = int(N/num_nodes)

    for i in range(num_nodes):
        if i == num_nodes - 1:
            data_idx = idxs[m*i:]
        else:
            data_idx = idxs[m*i:m*(i+1)]
        pos = tuple(random_gen.random() for k in range(2))
        data_matrix = np.hstack(tuple(data[data_idx]))
        num_examples = len(data_idx)
        code_dims = (n_components, data_matrix.shape[1])
        code_mat = np.random.rand(*code_dims)
        graph.add_node(i, pos=pos, num_examples=num_examples, data_matrix=data_matrix, code_matrix=code_mat)

    if topo == "geometric":
        graph.add_edges_from(nx.geometric_edges(graph, radius))

    assert nx.is_connected(graph)

    return graph


def create_optimizer(optim, W, args, X=None, H=None):
    if optim == "rmiso":
        return Rmiso(W, n_nodes=args.graph_size, n_components=args.n_components, alpha=args.alpha,
                     rho=args.rho, beta=args.beta, dynamic_reg=args.dynamic_reg)
    elif optim == "minibatch_cd":
        return MiniBatchCD(W, n_nodes=args.graph_size, n_components=args.n_components, alpha=args.alpha)
    elif optim == "mu":
        return MU(W, rho=0, delta=0, n_components=args.n_components, alpha=args.alpha, X=X, H=H)
    elif optim == "psgd":
        return PSGD(W, X=X, H=H, lr=args.lr, eta=args.eta, n_nodes=args.graph_size,
                    n_components=args.n_components, alpha=args.alpha)
    elif optim == "adagrad":
        return AdaGrad(W, lr=args.lr, n_components=args.n_components, alpha=args.alpha)
    else:
        raise ValueError("Invalid optimizer: {}".format(optim))


def get_save_name(args):
    name = {
        "rmiso": '{}-rho{:f}-beta{:f}-dynamic{}'.format(args.optim, args.rho, args.beta, args.dynamic_reg),
        'minibatch_cd': '{}'.format(args.optim),
        'mu': '{}'.format(args.optim),
        'psgd': '{}'.format(args.optim) if not args.full_batch else 'pgd',
        'adagrad': '{}-lr{}'.format(args.optim, args.lr)
    }[args.optim]
    return '{}-{}-alpha{}-components{}-nodes{}-{}'.format(args.dataset, name, args.alpha, args.n_components,
                                                     args.graph_size, args.sampling_algorithm)


def loss(X, W, H, alpha=0):
    A = X - np.dot(W, H)
    return 0.5*LA.norm(A, 'fro') + alpha*LA.norm(H, 1)


def evaluate(loss_fn, graph, W, alpha=0):
    eval_loss = 0
    n = len(graph.nodes)
    for node in range(n):
        X = graph.nodes[node]['data_matrix']
        H = graph.nodes[node]['code_matrix']
        n_v = graph.nodes[node]['num_examples']
        eval_loss += (1/n_v)*loss_fn(X, W, H, alpha=alpha)
    eval_loss *=1/n
    return eval_loss


def init_optimizer(optimizer, graph):
    assert isinstance(optimizer, Rmiso), "optimizer does not need to be initialized"
    start = time.time()
    for node in graph.nodes:
        optimizer.set_curr_node(node)
        X = graph.nodes[node]['data_matrix']
        optimizer.set_data_matrix(X)
        # be sure to set data matrix before initializing surrogate
        optimizer.init_surrogate()
        graph.nodes[node]['code_matrix'] = optimizer.get_code_matrix()
    end = time.time()
    return end - start


def train(optimizer, graph, sampler, loss_fn, n_iter=10, initial_time=0.0):
    losses = []
    elapsed_time = []
    alpha = optimizer.alpha
    for i in tqdm(range(n_iter)):
        start = time.time()
        node_id = sampler.step()
        X = graph.nodes[node_id]["data_matrix"]
        H = graph.nodes[node_id]["code_matrix"]
        if isinstance(optimizer, Rmiso):
            optimizer.set_curr_node(node_id)
        optimizer.set_data_matrix(X)
        optimizer.set_code_matrix(H)
        optimizer.step()
        end = time.time()
        graph.nodes[node_id]["code_matrix"] = optimizer.get_code_matrix()
        loss_val = evaluate(loss_fn, graph, optimizer.W, alpha)
        losses.append(loss_val)
        if i == 0:
            elapsed_time.append(end - start + initial_time)
        else:
            last_time = elapsed_time[-1]
            elapsed_time.append(last_time + end - start)

    return losses, elapsed_time


def train_full_batch(optimizer, loss_fn, n_iter=10, n_samples=1, initial_time=0.0):
    losses = []
    alpha = optimizer.alpha
    elapsed_time = []
    for i in tqdm(range(n_iter)):
        start = time.time()
        optimizer.step()
        end = time.time()
        loss_val = (1/n_samples)*loss_fn(optimizer.X, optimizer.W, optimizer.H, alpha=alpha)
        losses.append(loss_val)
        elapsed_time.append(end - start + initial_time)
        initial_time = end - start + initial_time
    return losses, elapsed_time


def main():
    parser = get_parser()
    args = parser.parse_args()

    np.random.seed(0)
    data, dims = build_dataset("mnist", num_examples=args.n_examples)
    m, n, d = dims[0], dims[1], args.n_components

    if args.full_batch:
        assert args.optim in FULL_BATCH_ALGS
        n *= args.n_examples
        X = build_full_batch(data)
        W = np.maximum(np.random.rand(m, d), np.zeros(shape=(m, d)))
        H = np.maximum(np.random.rand(d, n), np.zeros(shape=(d, n)))
        optimizer = create_optimizer(args.optim, W, args, X=X, H=H)
        losses, times = train_full_batch(optimizer, loss, n_iter=args.iterations, n_samples=args.n_examples)
    else:
        assert args.optim in MINI_BATCH_ALGS
        graph = build_graph(data, topo=args.graph_topo, num_nodes=args.graph_size, radius=args.radius,
                            n_components=args.n_components)
        sampler = {
            "uniform": Uniform,
            "metropolis_hastings": MetropolisHastings,
            "random_walk": RandomWalk
        }[args.sampling_algorithm](graph=graph)

        W = np.maximum(np.random.rand(m, d), np.zeros(shape=(m, d)))
        optimizer = create_optimizer(args.optim, W, args)
        if isinstance(optimizer, Rmiso):
            initialization_time = init_optimizer(optimizer, graph)
        else:
            initialization_time = 0
        losses, times = train(optimizer, graph, sampler, loss, n_iter=args.iterations, initial_time=initialization_time)

    save_name = get_save_name(args)
    if not os.path.isdir('curve'):
        os.mkdir('curve')

    result_dict = {"losses": losses, "times": times}
    path = os.path.join('curve', save_name)
    pickle.dump(result_dict, open(path, 'wb'))


if __name__ == "__main__":
    main()

