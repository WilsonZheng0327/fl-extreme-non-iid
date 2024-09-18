import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def load_dataset(download=True):
    # Data augmentation and normalization for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])

    # Just normalization for testing
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])

    # Loading the CIFAR-100 dataset
    # train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=download, transform=transform_train)
    # test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=download, transform=transform_test)

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=download, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=download, transform=transform_test)

    return train_dataset, test_dataset


def partition_dataset(config, train_dataset, test_dataset, logger):
    distribution = config.distribution
    num_clients = config.num_clients
    batch_size = config.batch_size
    alpha = config.alpha
    num_classes = config.num_classes

    if distribution == 'IID':
        logger.log('Creating IID partitions...')
        
        # Split the training dataset into subsets for each client
        indices = list(range(len(train_dataset)))
        subset_sizes = [len(train_dataset) // num_clients] * num_clients
        subset_sizes[-1] += len(train_dataset) % num_clients
        subsets = [Subset(train_dataset, indices[offset:offset+size]) for offset, size in zip(torch.cumsum(torch.tensor([0]+subset_sizes), 0), subset_sizes)]

        # Create DataLoaders for each client
        train_loaders = [DataLoader(subset, batch_size=batch_size, shuffle=True) for subset in subsets]
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    elif distribution == 'non-IID':
        logger.log('Creating non-IID partitions...')
                
        # Get the labels of the training dataset
        train_labels = np.array(train_dataset.targets)

        # Generate the label distribution for each client using the Dirichlet distribution
        if alpha == 0:
            # If alpha = 0, manually create extreme label distribution
            label_distribution = np.array([[1,0,0,0,0], [1,0,0,0,0], [0,1,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,1,0], [0,0,0,0,1], [0,0,0,0,1]])
            label_distribution = label_distribution.T
        else:
            max_diff = 0.2
            label_distribution = np.random.dirichlet(alpha=np.ones(num_clients)*alpha, size=num_classes)
            label_distribution = label_distribution.T
            distribution_sums = np.sum(label_distribution, axis=1)
            while max(distribution_sums) - min(distribution_sums) > max_diff:
                label_distribution = np.random.dirichlet(alpha=np.ones(num_clients)*alpha, size=num_classes)
                label_distribution = label_distribution.T
                distribution_sums = np.sum(label_distribution, axis=1)

        # Assign samples to clients based on the label distribution
        client_indices = [[] for _ in range(num_clients)]
        for i, label in enumerate(train_labels):
            client = np.random.choice(num_clients, p=label_distribution[:, label])
            client_indices[client].append(i)

        # Create subsets for each client based on the assigned indices
        subsets = [Subset(train_dataset, indices) for indices in client_indices]

        # Create DataLoaders for each client
        train_loaders = [DataLoader(subset, batch_size=batch_size, shuffle=True) for subset in subsets]
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # for train_loader in train_loaders:
    #     label_distribution = get_label_distribution(train_loader)
    #     print("Label Distribution:")
    #     for label, proportion in label_distribution.items():
    #         print(f"Label {label}: {proportion:.4f}")

    return train_loaders, test_loader

from collections import Counter

def get_label_distribution(dataloader):
    label_counts = Counter()
    
    for _, labels in dataloader:
        label_counts.update(labels.numpy())
    
    total_samples = sum(label_counts.values())
    label_distribution = {label: count / total_samples for label, count in label_counts.items()}
    
    return label_distribution