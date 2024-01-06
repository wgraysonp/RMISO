import numpy as np
import torch
import torch.backends.cudnn as cudnn
import networkx as nx
from sampling_algorithms import *
from tqdm import tqdm
import argparse
from sklearn.datasets import fetch_openml
from custom_optimizers import *
from model import NMF
import os
import time
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

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
    parser.add_argument('--n_components', default=15, type=int,
                        help='number of dictionary components (columns in the matrix)')
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
        X = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)[0]
        X = X / 255.0
        n = len(X)
        X = X.reshape(n, 28, 28)
        idx = random_gen.sample(range(n), num_examples)
        dataset = X[idx]
        #dataset = torch.from_numpy(dataset)
        dataset = torch.tensor(dataset, dtype=torch.float32)
        dims = (28, 28)

    return dataset, dims


def build_graph(data, num_nodes=10, radius=0.3, topo="geometric", n_components=15):
    assert topo in GRAPH_TOPOS, "invalid graph topology"
    assert len(data) >= num_nodes, "not enough nodes"

    random_gen = random.Random(4)
    graph = nx.Graph()
    N = len(data)
    idxs = list(range(N))
    m = int(N / num_nodes)

    for i in range(num_nodes):
        if i == num_nodes - 1:
            data_idx = idxs[m * i:]
        else:
            data_idx = idxs[m * i:m * (i + 1)]
        pos = tuple(random_gen.random() for k in range(2))
        data_matrix = torch.hstack(tuple(data[data_idx]))
        num_examples = len(data_idx)
        code_dims = (n_components, data_matrix.shape[1])
        # ensure initial guess is non-negative
        code_mat = torch.randn(code_dims).clamp_(min=0)
        graph.add_node(i, pos=pos, num_examples=num_examples, data_matrix=data_matrix, code_matrix=code_mat)

    if topo == "geometric":
        graph.add_edges_from(nx.geometric_edges(graph, radius))

    assert nx.is_connected(graph)

    return graph


def create_optimizer(params, args):
    optim = args.optim
    if optim == 'rmiso':
        return RMISOBCD(params, target_rows=28, n_components=args.n_components, n_nodes=args.graph_size,
                        dynamic_reg=args.dynamic_reg, rho=args.rho, beta=args.beta)
    elif optim == 'psgd':
        return PSGD(params, args.lr)
    elif optim == 'adagrad':
        return PAdagrad(params, args.lr)
    else:
        raise ValueError("Invalid Optimizer")


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


def build_model(args, device):
    W = torch.randn(28, args.n_components).clamp_(min=0)
    n = int(args.n_examples / args.graph_size)*28
    H = torch.randn(args.n_components, n).clamp_(min=0)
    model = NMF(W, H, n_components=args.n_components)

    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    return model


def init_optimizer(optimizer, criterion, model, device, graph, alpha=0):
    assert isinstance(optimizer, RMISOBCD), "optimizer does not need to be initialized"
    assert isinstance(model, NMF)
    start = time.time()
    for node in graph.nodes:
        optimizer.zero_grad()
        X = graph.nodes[node]['data_matrix']
        H = graph.nodes[node]['code_matrix']
        model.set_code_matrix(H)
        X = X.to(device)
        loss = criterion(X, model.forward(), H, alpha)
        loss.backward()
        closure = lambda: criterion(X, model.forward(), H, alpha)
        optimizer.set_data_node(node, X)
        optimizer.init_surrogate(closure=closure)
        graph.nodes[node]['code_matrix'] = model.get_code_matrix()
    end = time.time()
    return end - start


def loss_fn(X, Y, H, alpha):
    return 0.5*torch.norm(X - Y)**2 + alpha * torch.norm(H, p=1)


def evaluate(graph, model, alpha=0):
    eval_loss = 0
    n = len(graph.nodes)
    with torch.no_grad():
        for node in range(n):
            X = graph.nodes[node]['data_matrix']
            H = graph.nodes[node]['code_matrix']
            model.set_code_matrix(H)
            n_v = graph.nodes[node]['num_examples']
            eval_loss += (1/n_v)*loss_fn(X, model.forward(), H, alpha)
        eval_loss *= 1/n
    return eval_loss


def train(optimizer, graph, sampler, model, criterion, device, alpha=0, n_iter=10, initial_time=0.0):
    losses = []
    elapsed_time = []
    for i in tqdm(range(n_iter)):
        start = time.time()
        node = sampler.step()
        X = graph.nodes[node]['data_matrix']
        H = graph.nodes[node]['code_matrix']
        model.set_code_matrix(H)
        X = X.to(device)
        optimizer.zero_grad()
        loss = criterion(X, model.forward(), model.params.H.data, alpha)
        loss.backward()
        if isinstance(optimizer, RMISOBCD):
            closure = lambda: criterion(X, model.forward(), model.params.H.data, alpha)
        else:
            closure = None
        optimizer.step(closure=closure)
        end = time.time()
        graph.nodes[node]['code_matrix'] = model.get_code_matrix()
        eval_loss = evaluate(graph, model, alpha)
        losses.append(eval_loss)
        if i == 0:
            elapsed_time.append(end - start + initial_time)
        else:
            last_time = elapsed_time[-1]
            elapsed_time.append(last_time + end - start)

        if i % 100 == 0:
            print("train loss: {:3f}".format(eval_loss))

    return losses, elapsed_time


def main():
    parser = get_parser()
    args = parser.parse_args()

    torch.manual_seed(10)
    data, _ = build_dataset(args.dataset, num_examples=args.n_examples)

    graph = build_graph(data, num_nodes=args.graph_size, n_components=args.n_components, radius=args.radius,
                        topo=args.graph_topo)

    sampler = {
        "uniform": Uniform,
        "metropolis_hastings": MetropolisHastings,
        "random_walk": RandomWalk
    }[args.sampling_algorithm](graph=graph)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args, device)
    param_groups = [
        {'params': model.params.W, 'name': 'W'},
        {'params': model.params.H, 'name': 'H'}
    ]
    optimizer = create_optimizer(param_groups, args)
    if isinstance(optimizer, RMISOBCD):
        init_time = init_optimizer(optimizer, loss_fn, model, device, graph, alpha=args.alpha)
    else:
        init_time = 0
    losses, times = train(optimizer, graph, sampler, model, loss_fn,
                          device, alpha=args.alpha, n_iter=args.iterations, initial_time=init_time)

    #save_name = get_save_name(args)
    #if not os.path.isdir('curve')


if __name__ == "__main__":
    main()
