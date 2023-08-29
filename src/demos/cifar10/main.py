import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import argparse
import pickle

from models import *
from custom_optimizers import *
from adabound import AdaBound

from graph_structure.data_graph import DataGraph
from regularization_scheduler import RegScheduler


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--sampling_algorithm', default='uniform', type=str, help='algorithm to sample from graph',
                        choices=['uniform', 'metropolis_hastings'])
    parser.add_argument('--graph_size', default=100, type=int, help='number of nodes in the graph')
    parser.add_argument('--graph_edges', default=99, type=int, help='number of edges in the graph')
    parser.add_argument('--graph_topo', default='random', type=str, help='model',
                        choices=['random', 'cycle', 'geometric'])
    parser.add_argument('--radius', default=0.5, type=float,
                        help='connection radius for geometric graph on unit square')
    parser.add_argument('--model', default='resnet', type=str, help='model',
                        choices=['resnet', 'densenet'])
    parser.add_argument('--optim', default='rmiso', type=str, help='optimizer',
                        choices=['rmiso', 'sgd', 'adagrad', 'adam', 'amsgrad', 'adabound', 'amsbound', 'mcsag'])
    parser.add_argument('--lr', default=1, type=float, help='learning rate')
    parser.add_argument('--start_factor', default=1, type=float, help='start factor for linear LR scheduler')
    parser.add_argument('--total_sched_iters', default=1, type=float, help='last epoch in linear LR scheduler')
    parser.add_argument('--final_lr', default=0.1, type=float,
                        help='final learning rate of AdaBound')
    parser.add_argument('--reg_step', default=200, type=float, help='reg scheduler step size')
    parser.add_argument('--reg_gamma', default=1, type=float, help='reg scheduler growth rate')
    parser.add_argument('--gamma', default=1e-3, type=float,
                        help='convergence speed term of Adabound')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum term')
    parser.add_argument('--rho', default=1, type=float, help='rmiso proximal regularization parameter')
    parser.add_argument('--delta', default=1, type=float, help='rmiso dynamic reg multiplier')
    parser.add_argument('--pr_floor', default=0, type=float, help='set floor of rmiso pr parameter')
    parser.add_argument('--tau', default=1, type=float, help='mcsag hitting time. set to 1 for o.g. sag')
    parser.add_argument('--dynamic_step', action='store_true',
                        help='rmiso and mcsag dynamic lr schedule')
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam coefficients beta_1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam coefficients beta_2')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='weight decay for optimizers')
    parser.add_argument('--save_graph', action='store_true', help='save the data graph')
    parser.add_argument('--save', action='store_true', help='save training curve')
    return parser


def build_dataset(args):
    print('==> Preparing data..')
    transform_trane = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_trane)
    graph = DataGraph(train_set, num_nodes=args.graph_size, num_edges=args.graph_edges, topo=args.graph_topo,
                      radius=args.radius, algorithm=args.sampling_algorithm)

    if args.save_graph:
        directory = os.path.join(os.getcwd(), "saved_graphs")
        os.makedirs(directory, exist_ok=True)
        f_name = "data_graph-{}-{}-nodes{}-edges{}.pickle".format(args.sampling_algorithm, args.model,
                                                                  args.graph_size, args.graph_edges)
        path = os.path.join(directory, f_name)
        pickle.dump(graph, open(path, 'wb'))

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)

    return graph, test_loader


def get_ckpt_name(model='resnet', optimizer='sgd', lr=0.1, final_lr=0.1, momentum=0.9, beta1=0.9, beta2=0.999, gamma=1e-3,
                  rho=1, start_factor=1, total_iters=1, delta=1, graph_size=10, graph_edges=10, sampling_alg='uniform'):
    initial_lr = start_factor*lr
    final_lr = lr
    name = {
        'sgd': 'lr{}-momentum{}'.format(lr, momentum),
        'adagrad': 'lr{}'.format(lr),
        'adam': 'lr{}-betas{}-{}'.format(lr, beta1, beta2),
        'amsgrad': 'lr{}-betas{}-{}'.format(lr, beta1, beta2),
        'adabound': 'lr{}-betas{}-{}-final_lr{}-gamma{}'.format(lr, beta1, beta2, final_lr, gamma),
        'amsbound': 'lr{}-betas{}-{}-final_lr{}-gamma{}'.format(lr, beta1, beta2, final_lr, gamma),
        'rmiso': 'lr{}-rho{:f}-delta{:f}-initial_lr{:f}-final_lr{:f}-end{}'.format(lr, rho, delta, initial_lr, final_lr, total_iters),
        'mcsag': 'lr{:f}-rho{:f}-delta{:f}'.format(lr, rho, delta),
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
        'resnet': ResNet34,
        'densenet': DenseNet121
    }[args.model]()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if ckpt:
        net.load_stat_dict(ckpt['net'])

    return net


def create_optimizer(args, num_nodes, model_params):
    if args.optim == 'sgd':
        return optim.SGD(model_params, args.lr, momentum=args.momentum,
                         weight_decay=args.weight_decay)
    elif args.optim == 'adagrad':
        return optim.Adagrad(model_params, args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        return optim.Adam(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay)
    elif args.optim == 'amsgrad':
        return optim.Adam(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay, amsgrad=True)
    elif args.optim == 'rmiso':
        return RMISO(model_params, args.lr, num_nodes=num_nodes,
                     dynamic_step=args.dynamic_step, rho=args.rho,
                     delta=args.delta, pr_floor=args.pr_floor, weight_decay=args.weight_decay)
    elif args.optim == 'mcsag':
        return MCSAG(model_params, args.lr, num_nodes=num_nodes,
                     dynamic_step=args.dynamic_step, tau=args.tau, pr_floor=args.pr_floor,
                     delta=args.delta, weight_decay=args.weight_decay)
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
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    total = 0
    correct = 0
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
        if isinstance(optimizer, (RMISO, MCSAG)):
            optimizer.set_current_node(node_id)
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    print('acc {:.3f}'.format(accuracy))
    print('loss {:.3f}'.format(train_loss))

    return accuracy, train_loss


def evaluate(net, device, data_loader, criterion, data_set="test"):
    net.eval()
    eval_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            eval_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    print('{} acc {:.3f}'.format(data_set, accuracy))
    print('{} loss {:.3f}'.format(data_set, eval_loss))

    return accuracy, eval_loss


def main():
    parser = get_parser()
    args = parser.parse_args()

    graph, test_loader = build_dataset(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ckpt_name = get_ckpt_name(model=args.model, optimizer=args.optim, lr=args.lr,
                              final_lr=args.final_lr, momentum=args.momentum,
                              beta1=args.beta1, beta2=args.beta2, gamma=args.gamma, start_factor=args.start_factor,
                              total_iters=args.total_sched_iters, graph_size=args.graph_size, rho=args.rho, delta=args.delta,
                              graph_edges=args.graph_edges, sampling_alg=args.sampling_algorithm)

    if args.resume:
        ckpt = load_checkpoint(ckpt_name)
        best_acc = ckpt['acc']
        start_epoch = ckpt['epoch']
    else:
        ckpt = None
        best_acc = 0
        start_epoch = -1

    net = build_model(args, device, ckpt=ckpt)
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(args, args.graph_size, net.parameters())
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=args.start_factor,
                                            total_iters=args.total_sched_iters, verbose=True)

    train_accuracies = []
    test_accuracies = []
    train_losses = []
    test_losses = []

    for epoch in range(start_epoch + 1, 200):
        train_acc, train_loss =  train(net, epoch, device, graph, optimizer, criterion)
        test_acc, test_loss = evaluate(net, device, test_loader, criterion, data_set="test")
        #scheduler.step()
        if args.sampling_algorithm == "metropolis_hastings" and isinstance(optimizer, (RMISO, MCSAG)):
            for group in optimizer.param_groups:
                print("rho: {}".format(group['rho']))

        if args.save:
            # Save checkpoint.
            if test_acc > best_acc:
                print('Saving..')
                state = {
                    'net': net.state_dict(),
                    'acc': test_acc,
                    'epoch': epoch,
                }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save(state, os.path.join('checkpoint', ckpt_name))
                best_acc = test_acc

            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            if not os.path.isdir('curve'):
                os.mkdir('curve')
            torch.save({'train_acc': train_accuracies, 'test_acc': test_accuracies, 'train_loss': train_losses,
                        'test_loss': test_losses}, os.path.join('curve', ckpt_name))


if __name__ == "__main__":
    main()


