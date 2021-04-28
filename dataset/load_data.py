import torch
import torchvision
import torchvision.transforms as transforms
from config import args


def create_train_dataset(batch_size=128, root='./data'):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.class_num == 10:
        train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    elif args.class_num == 100:
        print('using cifar100')
        train_set = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform_train)
    else:
        raise NotImplementedError
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

    return train_loader


def create_test_dataset(batch_size=128, root='./data'):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.class_num == 10:
        test_set = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
    elif args.class_num == 100:
        test_set = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=transform_test)
    else:
        raise NotImplementedError
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)
    return test_loader
