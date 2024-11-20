import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import os
from torch.utils.data import DataLoader

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define custom data loader for our specific structure
def data_loader(data_dir,
                batch_size,
                random_seed=42,
                valid_size=0.1,
                shuffle=True,
                test=False):

    # If test flag is True load from test folder
    if test:
        # Assume you have a separate folder for testing images
        dataset = datasets.ImageFolder(
            root=os.path.join(data_dir, 'test')
        )

        data_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )

        return data_loader

    # Load the train part of the dataset
    full_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'train')
    )

    # Split the training and validation
    num_train = len(full_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    # Shuffle the data indices if shuffle is True
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # Create DataLoaders for training and validation
    train_loader = DataLoader(
        full_dataset, batch_size=batch_size, sampler=train_sampler
    )

    valid_loader = DataLoader(
        full_dataset, batch_size=batch_size, sampler=valid_sampler
    )

    return train_loader, valid_loader

# Examples of usage
train_loader, valid_loader = data_loader(data_dir=os.path.join(os.getcwd(), '../CompCars/data/split_cropped_image'),
                                         batch_size=64)

test_loader = data_loader(data_dir=os.path.join(os.getcwd(), '../CompCars/data/split_cropped_image'),
                          batch_size=64, test=True)
