"""Functions for dataset download and processing."""
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, Dataset, Subset, random_split
from torchvision.datasets import MNIST

from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor


def load_data():
    """Load CIFAR-10 (training and test set)."""
    trf = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    trainset = CIFAR10("./data", train=True, download=True, transform=trf)
    testset = CIFAR10("./data", train=False, download=True, transform=trf)

    num_examples = {"trainset": len(trainset), "testset": len(testset)}
    return trainset, testset, num_examples


def load_partition(num_clients: int, idx: int):
    """Load 1/50th of the training and test data to simulate a partition."""
    assert idx in range(num_clients)
    trainset, testset, num_examples = load_data()
    n_train = int(num_examples["trainset"] / num_clients)
    n_test = int(num_examples["testset"] / num_clients)

    train_partitions = [torch.utils.data.Subset(
        trainset, range(i * n_train, (i + 1) * n_train)
    ) for i in range(num_clients)]
    test_partition = torch.utils.data.Subset(
        testset, range(idx * n_test, (idx + 1) * n_test)
    )
    return train_partitions, test_partition

