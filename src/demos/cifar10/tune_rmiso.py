import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse

from models import *
from custom_optimizers.RMISO import RMISO
from custom_optimizers.MCSAG import MCSAG
from graph_structure.data_graph import DataGraph
from regularization_scheduler import RegScheduler


def get_parser():
    parser = argparse.ArgumentParser(description='Tune RMISO Hyperparameters on small portion of CIFAR10 data')
    parser.add_argument('--sampling_algorithm', default='uniform', type=str, help='algorithm to sample from graph',
                        choices=['uniform', 'metropolis_hastings'])
    parser.add_argument('--optim', default='rmiso', type=str, help='optimizer',
                        choices=['rmiso', 'sgd', 'mcsag', 'adam'])
    parser.add_argument('--model', default='resnet', type=str, help='model',
                        choices=['resnet', 'densenet'])
    parser.add_argument('--lr', default=1, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='sgd momentum')
    parser.add_argument('--rho', default=1, type=float, help='rmiso proximal regularization parameter')
    parser.add_argument('--dynamic_step', action='store_true',
                        help='rmiso dynamic proximal regularization schedule')
    parser.add_argument('--load_graph', action='store_true', help='load previously used graph')
    parser.add_argument('--init_rmiso', action='store_true',
                        help='do one loop over the training data to initialize rmiso gradients')
    parser.add_argument('--delta', type=float, help='rmiso reg')
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

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_trane)

    data_subset = list(range(5000))

    train_subset = torch.utils.data.Subset(trainset, data_subset)

    graph = DataGraph(train_subset, num_nodes=50, num_edges=190, algorithm=args.sampling_algorithm, topo='geometric', radius=0.3)
    return graph


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
        return optim.SGD(model_params, args.lr, momentum=args.momentum)
    elif args.optim == 'rmiso':
        return RMISO(model_params, args.lr, num_nodes=num_nodes,
                     dynamic_step=args.dynamic_step, rho=args.rho, delta=args.delta)
    elif args.optim == 'mcsag':
        return MCSAG(model_params, args.lr, num_nodes=num_nodes,
                     dynamic_step=args.dynamic_step, rho=args.rho)
    elif args.optim == 'adam':
        return optim.Adam(model_params, args.lr, betas=(0.9, 0.99))
    else:
        raise ValueError("invalid optimizer")


def initialize_optimizer(net, device, graph, optimizer, criterion):
    assert isinstance(optimizer, (RMISO, MCSAG))
    print("== initializing gradients")
    n_iter = len(graph.nodes)
    for i in tqdm(range(n_iter)):
        loader = graph.nodes[i]['loader']
        assert len(loader) == 1
        (inputs, targets) = next(iter(loader))
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.set_current_node(i)
        optimizer.init_params()


def train(net, epoch, device, graph, optimizer, criterion):
    print('\nEpoch: %d' % epoch)
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
        print("curr_node: {}".format(node_id))
        if isinstance(optimizer, (RMISO, MCSAG)):
            optimizer.set_current_node(node_id)
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    print('loss: {:3f}'.format(train_loss))
    print('train acc %.3f' % accuracy)

    return accuracy


def main():
    parser = get_parser()
    args = parser.parse_args()

    graph = build_dataset(args)
    num_nodes = len(graph.nodes)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    start_epoch = -1

    net = build_model(args, device, ckpt=None)
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(args, num_nodes, net.parameters())
    reg_scheduler = RegScheduler(optimizer, name='rho', stepsize=10, gamma=2, verbose=True)
    
    if args.init_rmiso:
        initialize_optimizer(net, device, graph, optimizer, criterion)

    train_accuracies = []

    for epoch in range(start_epoch + 1, 10):
        train_acc = train(net, epoch, device, graph, optimizer, criterion)
        #reg_scheduler.step()

        train_accuracies.append(train_acc)


if __name__ == "__main__":
    main()
