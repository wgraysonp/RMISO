import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import argparse

from models import *
from RMISO import RMISO
from adabound import AdaBound

from graph_structure.graph_sampler import GraphBatchSampler


def get_parser():
    parser = argparse.ArgumentParser(description='Tune RMISO Hyperparameters on small portion of CIFAR10 data')
    parser.add_argument('--sampling_algorithm', default='uniform', type=str, help='algorithm to sample from graph',
                        choices=['uniform', 'metropolis_hastings'])
    parser.add_argument('--model', default='resnet', type=str, help='model',
                        choices=['resnet', 'densenet'])
    parser.add_argument('--lr', default=1, type=float, help='learning rate')
    parser.add_argument('--rho', default=1, type=float, help='rmiso proximal regularization parameter')
    parser.add_argument('--dynamic_step', action='store_true',
                        help='rmiso dynamic proximal regularization schedule')
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

    data_subset = list(range(128*5))

    train_subset = torch.utils.data.Subset(trainset, data_subset)
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=False, num_workers=2)

    return train_loader


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


def create_optimizer(args, model_params):
    return RMISO(model_params, args.lr, batch_num=5,
                 dynamic_step=args.dynamic_step, rho=args.rho)


def train(net, epoch, device, data_loader, optimizer, criterion):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(tqdm(data_loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.set_current_node(batch_idx)
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct/total
    print('train acc %.3f' % accuracy)

    return accuracy


def main():
    parser = get_parser()
    args = parser.parse_args()

    train_loader = build_dataset(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    start_epoch = -1

    net = build_model(args, device, ckpt=None)
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(args, net.parameters())

    train_accuracies = []

    for epoch in range(start_epoch + 1, 20):
        train_acc = train(net, epoch, device, train_loader, optimizer, criterion)

        train_accuracies.append(train_acc)


if __name__ == "__main__":
    main()