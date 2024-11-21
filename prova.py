import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import os
from torch.utils.data import DataLoader
import custom_models as cst
import gc
from tqdm import tqdm

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define custom data loader for our specific structure
def data_loader(data_dir,
                batch_size,
                random_seed=42,
                valid_size=0.1,
                shuffle=True,
                test=False):

    normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )

    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize,
    ])

    # If test flag is True load from test folder
    if test:
        # Assume you have a separate folder for testing images
        dataset = datasets.ImageFolder(
            root=os.path.join(data_dir, 'test'),
            transform=transform
        )

        data_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )

        return data_loader

    # Load the train part of the dataset
    full_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'train'),
        transform=transform
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

    return (train_loader, valid_loader)

# Define parameters
num_classes = 163
num_epochs = 2
batch_size = 64
learning_rate = 0.01

# Creating the loaders
train_loader, valid_loader = data_loader(data_dir=os.path.join(os.getcwd(), '../CompCars/data/split_cropped_image1'),
                                         batch_size=batch_size)

#test_loader = data_loader(data_dir=os.path.join(os.getcwd(), '../CompCars/data/split_cropped_image'),
#                          batch_size=batch_size, test=True)

model = cst.ResNet(cst.ResidualBlock, [2, 2, 2, 2],num_classes=num_classes).to(device)

# Loss and optimizer definition
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(),lr=learning_rate,weight_decay=0.001)  

for epoch in tqdm(range(num_epochs)):
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for i, (images, labels) in tqdm(enumerate(train_loader),desc=f'Currently running epoch number {epoch+1}',leave=True):  
        #Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        #Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        #Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

        del images, labels, outputs
        torch.cuda.empty_cache()
        gc.collect()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader)}, Accuracy: {100 * correct_train / total_train}%')


# Validation
 
model.eval()  # Set model to evaluation mode
correct_val = 0
total_val = 0

with torch.no_grad():
    for images, labels in tqdm(valid_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total_val += labels.size(0)
        correct_val += (predicted == labels).sum().item()
        del images, labels, outputs

    print(f'Accuracy of the network on the {len(valid_loader)} batches of validation images: {100 * correct_val / total_val} %')

torch.save(model.state_dict(),os.path.join(os.getcwd(),'../CompCars/data/model.pt'))