import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import os
from torch.utils.data import Dataset,DataLoader,Subset
import custom_models as cst
import gc
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, file_paths, transform=None, train=True, validation_split=0.1, random_seed=42, classification_type='make'):
        """
        Args:
            root_dir (str): Root directory containing the image files.
            file_paths (str): Path to the text file containing image paths (relative to root_dir).
            transform (callable, optional): Transform to be applied on an image.
            train (bool): If True, splits data into training and validation sets.
            validation_split (float): Proportion of data to use for validation.
            random_seed (int): Random seed for reproducibility of splits.
            classification_type (str): make for car make classification, model for model classification
        """
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        
        if classification_type=='make':
            label_position=0
        elif classification_type=='model':
            label_position=1
        else:
            print('Wrong classification type') 

        # Load paths and labels from the text file
        with open(file_paths, 'r') as f:
            for line in f:
                path = line.strip()
                label = path.split('/')[label_position]
                self.image_paths.append(path)
                self.labels.append(int(label))  # Convert label to integer
                
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.labels)

        # If train flag is true, split into train and validation sets
        if train:
            train_indices, val_indices = train_test_split(
                range(len(self.image_paths)),
                test_size=validation_split,
                random_state=random_seed
            )
            self.train_indices = train_indices
            self.val_indices = val_indices

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and label
        img_path = os.path.join(self.root_dir, self.image_paths[idx])
        image = Image.open(img_path).convert("RGB")  # Ensure 3-channel RGB format
        label = self.labels[idx]

        # Apply transformations
        if self.transform:
            image = self.transform(image)
        return image, label
    


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define parameters
classification_type='model'  #make or model
num_epochs = 10
batch_size = 64
learning_rate = 1e-4
if classification_type=='make':
    num_classes=163
elif classification_type=='model':
    num_classes=1712
else:
    print('Wrong classification type') 

root_dir=os.path.join(os.getcwd(),'../CompCars/data/cropped_image')
file_paths_train=os.path.join(os.getcwd(),'../CompCars/data/train_test_split2/classification/train.txt')

dataset = CustomImageDataset(root_dir=root_dir, file_paths=file_paths_train, transform=transform, train=True, classification_type=classification_type)

# Create training and validation subsets
train_subset = Subset(dataset, dataset.train_indices)
val_subset = Subset(dataset, dataset.val_indices)

# Create dataloaders
train_loader = DataLoader(train_subset, batch_size=batch_size)
valid_loader = DataLoader(val_subset, batch_size=batch_size)

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
    i=0
    for images, labels in tqdm(train_loader,desc=f'Currently running epoch number {epoch+1}',leave=False):  
        #Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        #Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        i+=1
        running_loss+=loss.item() #the first one is wrong

        #Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predicted = torch.argmax(outputs.data, 1)
        total_train += outputs.size(0)
        correct_train += (predicted == labels).sum().item()

        del images, labels, outputs
        torch.cuda.empty_cache()
        gc.collect()

    # Validation
 
    model.eval()  # Set model to evaluation mode
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for images, labels in tqdm(valid_loader,leave=True):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predicted = torch.argmax(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
            del images, labels, outputs

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / i}, Training Accuracy: {100 * correct_train / total_train}%, Validation Accuracy: {100 * correct_val / total_val}%')


torch.save(model.state_dict(),os.path.join(os.getcwd(),'model_res.pt'))