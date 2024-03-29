import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

import os

"""
Functions:
    get_dataset
    train_val_split
    create_dataloader
"""

def get_dataset(dataset, path, train, transform):
    """
    Return torch Dataset object for DataLoader
    """
    print(f"Creating Dataloaders for {dataset}")
    if dataset == 'mnist':
        return torchvision.datasets.MNIST(
            root=path, train=train, transform=transform, download=True
            )
    elif dataset == 'cifar10':
        return torchvision.datasets.CIFAR10(
            root=path, train=train, transform=transform, download=True
        )
    elif dataset == 'cifar100':
        return torchvision.datasets.CIFAR100(
            root=path, train=train, transform=transform, download=True
        )
    elif dataset == 'svhn':
        if train:
            return torchvision.datasets.SVHN(
                root=path, split='train', transform=transform, download=True
            )
        else:
            return torchvision.datasets.SVHN(
                root=path, split='test', transform=transform, download=True
            )
    elif dataset == 'flowers102':
        if train:
            return torchvision.datasets.Flowers102(
                root=path, split='train', transform=transform, download=True
            )
        else:
            return torchvision.datasets.Flowers102(
                root=path, split='test', transform=transform, download=True
            )
    elif dataset == 'tiny_imagenet':
        if train:
            return get_tiny_imagenet(
                root=path, split='train', transform=transform
            )
        else:
            print('tiny_imagenet test is not implemented')
            return None
    else:
        raise NotImplementedError(f'Dataset {dataset} not implemented')

def get_tiny_imagenet(root, split, transform):
    """
    Return tiny imagenet dataset. Test split will return a None object
    """
    if split == 'test':
        return None
    elif split == 'train':
        
        full_path = os.path.join(root, 'tiny-imagenet-200', 'train')
        # check if this path exist or not
        if not os.path.exists(full_path):
            raise FileNotFoundError('tiny imagenet is missing')
        
        dataset = datasets.ImageFolder(root=full_path, transform=transform)
        return dataset

def train_val_split(train_dataset, val_split=0.2):
    """
    Split train dataset into train and validation sets
    """
    train_size = int((1 - val_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    return train_dataset, val_dataset
    
def get_dataloaders(config):
    """
    Return train, validation and test dataloaders
    """
    print('Creating dataloaders...')

    config_data = config['data']
    dataset = config_data['dataset']
    path = config_data['path']
    batch_size = config_data['batch_size']
    val_split = config_data['val_split']
    num_workers = config_data['num_workers']
    torch.manual_seed(42) # for reproducibility

    if dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)) # MNIST mean and std
            ])
    elif dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])
    elif dataset == 'cifar100':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        ])
    elif dataset == 'svhn':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4376821, 0.4437697, 0.47280442], std=[0.19803012, 0.20101562, 0.19703614])
        ])
    elif dataset == 'flowers102':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif dataset == 'tiny-imagenet':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # the statistics for imagenet
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    
    train_dataset = get_dataset(dataset, path, train=True, transform=transform)
    test_dataset = get_dataset(dataset, path, train=False, transform=transform)

    train_dataset, val_dataset = train_val_split(train_dataset, val_split=val_split)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
    if test_dataset:
        test_dataloader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
            )
    else:
        test_dataloader = None
    
    print('Dataloaders created')
    
    return train_dataloader, val_dataloader, test_dataloader