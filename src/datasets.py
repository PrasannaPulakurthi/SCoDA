"""
Dataset utilities for paired augmentations on MNIST and SVHN.
This adapts your MNISTPair / SVHNPair to a dedicated module.
"""

from typing import Tuple
from PIL import Image
from torchvision import transforms
from torchvision.datasets import MNIST, SVHN
import numpy as np
import torch


class MNISTPair(MNIST):
    """
    MNIST returning two transformed views (train and test transforms) + label.
    """

    def __init__(self, *args, test_transform=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_transform = test_transform

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy())

        pos_train = self.transform(img) if self.transform is not None else img
        pos_test = self.test_transform(img) if self.test_transform is not None else img

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_train, pos_test, target


class SVHNPair(SVHN):
    """
    SVHN returning two transformed views (train and test transforms) + label.
    """

    def __init__(self, *args, test_transform=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_transform = test_transform

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        img, target = self.data[index], int(self.labels[index])
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        pos_train = self.transform(img) if self.transform is not None else img
        pos_test = self.test_transform(img) if self.test_transform is not None else img

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_train, pos_test, target


# Transforms (kept close to your digest, but tidied up)
train_transform_mnist = transforms.Compose([
    transforms.RandomResizedCrop(28),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
])

test_transform_mnist = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
])

train_transform_svhn = transforms.Compose([
    transforms.RandomResizedCrop(28),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
])

test_transform_svhn = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
])
