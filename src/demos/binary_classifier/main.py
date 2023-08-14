import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse
import pickle
import sys

from adabound import AdaBound
from models import TwoLayer, OneLayer
from custom_optimizers import *
from graph_structure.data_graph import DataGraph
from datasets import CovType


def get_parser():
    parser = argparse.ArgumentParser(description="Binary Classifier Training")
    parser.add_argument('--sampling_algorithm', default='uniform', type=str, help='algorithm to sample from graph',
                        choices=['uniform', 'metropolis_hastings'])
    parser.add_argument('--graph_size', default=100, type=int, help='number of nodes in the graph')
    parser.add_argument('--graph_edges', default=99, type=int, help='number of edges in the graph')
    parser.add_argument('--model', default='one_layer', type=str, help='model',
                        choices=['one_layer', 'two_layer'])
    parser.add_argument('--optim', default='rmiso', type=str, help='optimizer',
                        choices=['rmiso', 'mcsag', 'sgd', 'adagrad', 'adam', 'amsgrad', 'adabound', 'amsbound'])
    parser.add_argument('--lr', default=1, type=float, help='learning rate')
    parser.add_argument('--final_lr', default=0.1, type=float,
                        help='final learning rate of AdaBound')
    parser.add_argument('--gamma', default=1e-3, type=float,
                        help='convergence speed term of Adabound')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum term')
    parser.add_argument('--rho', default=1, type=float, help='rmiso proximal regularization parameter')
    parser.add_argument('--delta', default=1e-5, type=float, help='rmiso dynamic prox reg multiplier')
    parser.add_argument('--tau', default=1, type=float, help='mcsag hitting time. set to 1 for o.g. sag')
    parser.add_argument('--dynamic_step', action='store_true',
                        help='rmiso and mcsag dynamic lr schedule')
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam coefficients beta_1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam coefficients beta_2')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='weight decay for optimizers')
    parser.add_argument('--load_graph', action='store_true', help='load previously used graph')
    parser.add_argument('--save', action='store_true', help='save training curve')
    return parser


def build_dataset(args):
    print('== Preparing data..')

    directory = os.path.join(os.getcwd(), "saved_graphs")
    os.makedirs(directory, exist_ok=True)
    f_name = "data_graph-{}-{}.pickle".format(args.sampling_algorithm, args.model)
    path = os.path.join(directory, f_name)

    zero_one = True if args.model == "one_layer" else False

    if args.load_graph:
        try:
            graph = pickle.load(open(path, 'rb'))
            assert isinstance(graph, DataGraph), "Graph loaded is not the correct object."
            print("Loading Graph-over-riding graph topology arguments")
        except FileNotFoundError:
            print("No graph available to load")
            sys.exit(1)
    else:
        train_set = CovType(train=True, zero_one=zero_one)
        graph = DataGraph(train_set, num_nodes=args.graph_size, num_edges=args.graph_edges,
                          algorithm=args.sampling_algorithm)
        if os.path.exists(path):
            os.remove(path)
        pickle.dump(graph, open(path, 'wb'))    

    test_set = CovType(train=False, zero_one=zero_one)
    test_loader = DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)

    return graph, test_loader


def get_ckpt_name(model='resnet', optimizer='sgd', lr=1e-3, final_lr=1e-3, momentum=0.9, beta1=0.9, beta2=0.999,
                  gamma=1e-3, rho=1, graph_size=10, graph_edges=10, sampling_alg='uniform'):
    name = {
        'sgd': 'lr{}-momentum{}'.format(lr, momentum),
        'adagrad': 'lr{}'.format(lr),
        'adam': 'lr{}-betas{}-{}'.format(lr, beta1, beta2),
        'amsgrad': 'lr{}-betas{}-{}'.format(lr, beta1, beta2),
        'adabound': 'lr{}-betas{}-{}-final_lr{}-gamma{}'.format(lr, beta1, beta2, final_lr, gamma),
        'amsbound': 'lr{}-betas{}-{}-final_lr{}-gamma{}'.format(lr, beta1, beta2, final_lr, gamma),
        'rmiso': 'lr{}-rho{}'.format(lr, rho),
        'mcsag': 'lr{}-rho{}'.format(lr, rho),
    }[optimizer]

    return '{}-{}-{}-nodes{}-edges{}-{}'.format(model, optimizer, name, graph_size, graph_edges, sampling_alg)


def load_checkpoint(ckpt_name):
    print('==> Resuming from checkpoint..')
    path = os.path.join('checkpoint', ckpt_name)
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found'
    assert os.path.exists(path), 'Error: checkpoint {} not found'.format(ckpt_name)
    return torch.load(ckpt_name)


def build_model(args, device, ckpt=None):
    print('==> Building model..')
    net = {
        'one_layer': OneLayer,
        'two_layer': TwoLayer
    }[args.model]()
    net = net.to(device)
    if device == 'cuda':
       # net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if ckpt:
        net.load_stat_dict(ckpt['net'])

    return net


def create_optimizer(args, num_nodes, model_params):
    if args.optim == 'sgd':
        return optim.SGD(model_params, args.lr, momentum=args.momentum,
                         weight_decay=args.weight_decay)
    elif args.optim == 'adagrad':
        return optim.Adam(model_params, args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        return optim.Adam(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay)
    elif args.optim == 'amsgrad':
        return optim.Adam(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay, amsgrad=True)
    elif args.optim == 'rmiso':
        return RMISO(model_params, args.lr, num_nodes=num_nodes,
                     dynamic_step=args.dynamic_step, rho=args.rho, delta=args.delta)
    elif args.optim == 'mcsag':
        return MCSAG(model_params, args.lr, num_nodes=num_nodes,
                     dynamic_step=args.dynamic_step, tau=args.tau, rho=args.rho)
    elif args.optim == 'adabound':
        return AdaBound(model_params, args.lr, betas=(args.beta1, args.beta2),
                        final_lr=args.final_lr, gamma=args.gamma,
                        weight_decay=args.weight_decay)
    else:
        assert args.optim == 'amsbound'
        return AdaBound(model_params, args.lr, betas=(args.beta1, args.beta2),
                        final_lr=args.final_lr, gamma=args.gamma,
                        weight_decay=args.weight_decay, amsbound=True)


def train(net, epoch, device, graph, optimizer, criterion):
    print('\nEpoch: %d' %  epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    n_iter = len(graph.nodes)

    for _ in tqdm(range(n_iter)):
        node_id = graph.sample()
        loader = graph.nodes[node_id]['loader']
        assert len(loader) == 1, f"Data loader at node {node_id} has more than one batch"
        (inputs, targets) = next(iter(loader))
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        if isinstance(optimizer, RMISO):
            optimizer.set_current_node(node_id)
        optimizer.step()
        train_loss += loss.item()/n_iter
        predicted = (outputs > 0.5).float() if isinstance(net, OneLayer) else (outputs > 0.0).float() - (outputs <= 0.0).float()
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    print('loss: {:3f}'.format(train_loss))
    print('train acc %.3f' % accuracy)

    return accuracy, train_loss


def test(net, device, data_loader, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    n_iter = len(data_loader)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()/n_iter
            predicted = (outputs > 0.5).float() if isinstance(net, OneLayer) else (outputs > 0.0).float() - (outputs <= 0.0).float()
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    print('test acc %.3f' % accuracy)

    return accuracy, test_loss


def main():
    parser = get_parser()
    args = parser.parse_args()

    graph_loader, test_loader = build_dataset(args)
    num_nodes = len(graph_loader.nodes)
    num_edges = len(graph_loader.edges)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ckpt_name = get_ckpt_name(model=args.model, optimizer=args.optim, lr=args.lr,
                              final_lr=args.final_lr, momentum=args.momentum,
                              beta1=args.beta1, beta2=args.beta2, gamma=args.gamma,
                              graph_size=num_nodes, rho=args.rho, graph_edges=num_edges,
                              sampling_alg=args.sampling_algorithm)

    if args.resume:
        ckpt = load_checkpoint(ckpt_name)
        best_acc = ckpt['acc']
        start_epoch = ckpt['epoch']
    else:
        ckpt = None
        best_acc = 0
        start_epoch = -1

    net = build_model(args, device, ckpt=ckpt)
    criterion = nn.MSELoss() if args.model == "one_layer" else nn.SoftMarginLoss()
    optimizer = create_optimizer(args, num_nodes, net.parameters())
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1, last_epoch=start_epoch)

    train_accuracies = []
    test_accuracies = []
    train_losses = []
    test_losses = []

    for epoch in range(start_epoch + 1, 50):
        train_acc, train_loss = train(net, epoch, device, graph_loader, optimizer, criterion)
        test_acc, test_loss = test(net, device, test_loader, criterion)
       # scheduler.step()

        if args.save:
         # Save checkpoint
            if test_acc > best_acc:
                print('Saving...')
                state = {
                    'net': net.state_dict(),
                    'acc': test_acc,
                    'loss': test_loss,
                    'epoch': epoch,
                 }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save(state, os.path.join('checkpoint', ckpt_name))
                best_acc = test_acc

            train_accuracies.append(train_acc)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            test_accuracies.append(test_acc)
            if not os.path.isdir('curve'):
                os.mkdir('curve')
            torch.save({'train_acc': train_accuracies, 'train_loss': train_losses, 'test_acc': test_accuracies, 'test_loss': test_losses},
                        os.path.join('curve', ckpt_name))


if __name__ == "__main__":
    main()
