import torch
from torchsummary import summary
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np 
from torchvision import transforms
import scipy.io
import sys
import cv2
from tqdm import tqdm

#PARAMETERS SECTION

classification_type='make'

#LABEL IMPORT SECTION

"""
types = scipy.io.loadmat(os.path.join(os.getcwd(),'CompCars/data/misc/car_type.mat'),squeeze_me=True)
types = types['types'].tolist()
#print(types)

make_model = scipy.io.loadmat(os.path.join(os.getcwd(),'CompCars/data/misc/make_model_name.mat'),squeeze_me=True)
makes = make_model['make_names'].tolist()
raw_models = make_model['model_names'].tolist()

filter_ids = filter(lambda i: isinstance(raw_models[i], str), range(len(raw_models)))

models = [raw_models[i] for i in filter_ids]
"""

#DATA IMPORT
"""
root_path=os.path.join(os.getcwd(),'CompCars/data')

images=[]
labels=[]

for subdir, dirs, files in (os.walk(os.path.join(root_path,'cropped_image'))):

    for file in files:

        image_path=os.path.join(subdir,file)
        image = cv2.imread(image_path)
        make, model, year, filename = image_path[image_path.find('image')+6:].split('/')

        images.append(image)

        if classification_type=='make':
            labels.append(make)
            NUM_CLASSES=163
        elif classification_type=='model':
            labels.append(model)
            NUM_CLASSES=1716

print(NUM_CLASSES)
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define a transformation to apply to the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to a fixed size
    transforms.ToTensor(),          # Convert images to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Set the path to the root directory containing your car makes
data_dir = os.path.join(os.getcwd(),'CompCars/data/cropped_image')

# Load the dataset using ImageFolder
train_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Create a DataLoader for batching and shuffling
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# The labels are inferred from the folder names
class_names = train_dataset.classes  # List of car makes
print(f"Car makes (labels): {class_names}")
print(len(class_names))

#MODEL CREATION SECTION

NUM_CLASSES=len(class_names)

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

summary(model, input_size=(3, 224, 224))

# Modify the fully connected (fc) layer to output 200 classes
model.fc = torch.nn.Linear(in_features=512, out_features=200)
model = model.to(device)
summary(model, input_size=(3, 224, 224))

# Now create a new model that includes ResNet and your additional layers.
class CustomResNet(torch.nn.Module):
    def __init__(self, resnet_model, num_classes):
        super(CustomResNet, self).__init__()
        self.resnet = resnet_model  # ResNet backbone
        self.relu = torch.nn.ReLU()      # ReLU after the ResNet output
        self.fc2 = torch.nn.Linear(200, num_classes)  # Your second fully connected layer
        self.softmax = torch.nn.Softmax(dim=1)  # Softmax at the end

    def forward(self, x):
        x = self.resnet(x)          # Pass through ResNet
        x = self.relu(x)            # Apply ReLU
        x = self.fc2(x)             # Apply second fully connected layer
        x = self.softmax(x)         # Apply Softmax
        return x

# Create the new custom model
custom_model = CustomResNet(model, num_classes=NUM_CLASSES)

for param in custom_model.resnet.parameters():
    param.requires_grad = False
custom_model.resnet.fc.requires_grad = True
print(custom_model.resnet.fc.parameters().requires_grad)

# Only the new layers should have requires_grad = True by default, 
# so you don't need to change them.

# Check which layers will be updated during training
for name, param in custom_model.named_parameters():
    if param.requires_grad:
        print(f'{name} will be trained.')
    else:
        print(f'{name} will not be trained.')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
custom_model = custom_model.to(device)

# Check the model summary
from torchsummary import summary
summary(custom_model, input_size=(3, 224, 224))


for param in model.parameters():
    param.requires_grad = False

# Now, only the newly added layers (Linear layers, ReLU, Softmax) will have their weights updated
for param in custom_model[-5:].parameters():  # Adjust if your layers are not at the very end
    param.requires_grad = True

criterion=torch.nn.CrossEntropyLoss()

optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

summary(model,input_size=(3,224,224))
summary(custom_model[-5:],input_size=(512,1,1))

from tqdm import tqdm

# Training loop
num_epochs = 1

for epoch in tqdm(range(num_epochs), desc="Epochs"):  # Wrap outer loop for epochs
    model.train()  # Set the model to training mode
    running_loss = 0.0

    # Wrap the DataLoader with tqdm for batch progress
    for inputs, labels in tqdm(train_loader, desc="Batches", leave=False):  # Wrap DataLoader for batches
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optim.zero_grad()

        # Forward pass
        outputs = model(inputs)
        
        # Compute the loss
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optim.step()

        running_loss += loss.item()

    # Print the average loss for this epoch
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")


torch.save(model.state_dict(),os.path.join(os.getcwd(),'CompCars/data/model.pt'))


"""
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
# Tensor of shape 1000, with confidence scores over ImageNet's 1000 classes
#print(output[0])

probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)

with open(os.path.join(os.getcwd(),"imagenet_classes.txt"), "r") as f:
    categories = [s.strip() for s in f.readlines()]
# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())
"""
"""
image_path=os.path.join(os.getcwd(),'CompCars/data/cropped_image')
#print(image_path)
dic={}

for subdir, dirs, files in os.walk(image_path):
    for file in files:

        file_path=os.path.join(subdir,file)

        image = Image.open(file_path)

        image_format = image.format
        image_size = image.size
        image_mode = image.mode
        image.close()

        if image_size in dic:
            dic[image_size]+=1
        else:
            dic[image_size]=1

maxcol=0
maxrow=0

minrow=10000
mincol=10000

max_app_val=0
max_app_key=()

for key in dic:
    if dic[key]>max_app_val:
        max_app_val=dic[key]
        max_app_key=key
    if key[0]>maxcol:
        maxcol=key[0]
    if key[1]>maxrow:
        maxrow=key[1]
    if key[0]<mincol:
        mincol=key[0]
    if key[1]<minrow:
        minrow=key[1]



print('dimension frequencies:',dic)
print('number of different frequencies:',len(dic))
print('maximum height of an image:',maxrow)
print('maximum width of an image:',maxcol)
print('minimum height of an image:',minrow)
print('minimum width of an image:',mincol)
print('most frequent dimension of an image:',max_app_key)
print('which appears ',max_app_val,' times')
"""