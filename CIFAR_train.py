'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torchvision.models.resnet import Bottleneck, BasicBlock
import argparse
import os

from prune_new import PruneTool
from prune import OldPruneTool


def get_argparse():
    parser = argparse.ArgumentParser('prune model for CIFAR10')

    # for model
    parser.add_argument('--num_classes', default=10)
    parser.add_argument('--pretrain', action='store true')
    parser.add_argument('--percentage', default=0.2, help='prune rate')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--block_type', choices=['basicblock', 'bottleneck'], required=True, help='can only prune such block type')

    # for data
    parser.add_argument('--data', default='./data', help='data path')
    parser.add_argument('--batch_size', default=256)

    # for optim
    parser.add_argument('--lr', default=0.1)
    parser.add_argument('--lr_decay', default=100)

    # for train and test
    parser.add_argument('--epoch', default=100)
    parser.add_argument('--device', default='cuda')

    args = parser.parse_args()
    return args


def testCIFAR10(model: nn.Module, test_loader: DataLoader, model_name=" ", device='cpu'):
    model.eval()
    correct = 0
    total_labels = 0
    with torch.no_grad():
        for (inputs, labels) in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            total_labels += labels.size()[0]
            output = model(inputs)
            _, pred = torch.max(output.data, 1)
            correct += (pred == labels).sum().item()
    print("model: {}\taccuracy: {}%".format(model_name, correct * 100 / total_labels))


def build_model(args):
    raw_model = resnet50(pretrained=args.pretrain, num_classes=args.num_classes).to(args.device)
    return raw_model


def trainCIFAR10(args):
    epoch = args.epoch
    percentage = args.percentage
    verbose = args.verbose
    device = args.device
    block_type = Bottleneck if args.block_type == 'bottleneck' else BasicBlock

    raw_model = build_model(args)

    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4), transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    train_set = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_set = torchvision.datasets.CIFAR10(root=args.data, train=False, download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
    optimizer = optim.SGD(raw_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.lr_decay)

    prune_tool = PruneTool(percentage, raw_model, device, block=block_type)

    for cur_epoch in range(epoch):
        raw_model.train()

        for i, data in enumerate(train_loader):
            optimizer.zero_grad()

            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            output = raw_model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            if i % 200 == 0:
                print("epoch: {}\titer: {}\tlr: {}\tloss: {}".format(cur_epoch + 1, i, scheduler.get_lr()[0], loss))
        scheduler.step()
        testCIFAR10(raw_model, test_loader, 'raw_model', device)

        # prune_tool = PruneTool(percentage, raw_model, device, block=block_type)
        prune_tool.reset_model(raw_model)
        prune_tool.mask_model_for_prune()
        prune_model = prune_tool.get_prune_model().to(device)
        compact_model = prune_tool.get_compact_model(verbose=verbose).to(device)

        testCIFAR10(prune_model, test_loader, 'prune_model', device)
        testCIFAR10(compact_model, test_loader, 'compact_model', device)

        save_model(raw_model, 'raw_model')
        save_model(prune_model, 'prune_model')
        save_model(compact_model, 'compact_model')

    print('train and test finished......')


def save_model(model, model_name, pre_path='./'):
    model_path = os.path.join(pre_path, model_name + ".pth")
    if os.path.exists(model_path):
        os.remove(model_path)

    torch.save(model, model_path, _use_new_zipfile_serialization=False)
    print('model: {} saved in {}'.format(model_name, pre_path))


def load_and_test(model_path, device='cpu'):
    model = torch.load(model_path, map_location='cpu')
    model = model.to(device)

    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    test_set = torchvision.datasets.CIFAR10(root=args.data, train=False, download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
    testCIFAR10(model, test_loader, 'model', device)


def main(args):
    trainCIFAR10(args)


if __name__ == '__main__':
    args = get_argparse()
    main(args)
