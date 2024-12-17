import custom_models as cst
import auxiliaries as aux
import os 
import torch
import torch.nn as nn
from torchvision import transforms
import os
from torch.utils.data import DataLoader, Subset
import gc
from tqdm import tqdm
from torchinfo import summary
import time

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

# Define parameters for current training
num_epochs = 10
batch_size = 64
learning_rate = 1e-4
patience = 4    # For early stopping
progressive = 1 # Used for differentiating between runs with same parameters
use_data_augmentation=True
splits_folder='train_test_split_verification_full_make_80'  # Folder which contains two files train.txt and test.txt with a list of images to use for train and test
classification_threshold=0.5
loss_name='binary_cross_entropy'

# Extract parameters from the extractor model name to be able to import it
extractor_model_save_name='model_inceptionmodified_make_8_focal_5.pt'
extractor_model_name,classification_type,_,_=extractor_model_save_name.split('.')[0].split('_')[1:-1]  # Extract parameters of the model to reconstruct it
image_type=splits_folder.split('_')[4]

# Name the file where we save the model with the parameters used for better readability
model_save_name=f'model_verification_{extractor_model_name}_{classification_type}_{batch_size}_{progressive}.pt'

# Define root path (where the images are) and train path (where the list of images to use is)
root_dir = os.path.join(os.getcwd(), f'../CompCars/data/{'cropped_image' if image_type=='full' else 'part'}')
file_paths_train = os.path.join(os.getcwd(), f'../CompCars/data/splits/{splits_folder}/verification/train.txt')

# Set number of classes
if classification_type == 'make':
    if image_type=='full':
        num_classes = 163
    elif image_type=='part':
        num_classes = 123
elif classification_type == 'model':
    if image_type == 'full':    
        num_classes = 1716
    elif image_type == 'part':
        num_classes = 956
else:
    print('Wrong classification type') 

# Create model
if extractor_model_name=='inceptionv1':
    model = cst.InceptionV1(in_channels=3, num_classes=num_classes).to(device)
elif extractor_model_name=='inceptionmodified':
    model = cst.InceptionModified(in_channels=3, num_classes=num_classes).to(device)
elif extractor_model_name=='resnet18':
    model = cst.ResNet(cst.ResidualBlock, [2, 2, 2, 2],num_classes=num_classes).to(device)
elif extractor_model_name=='resnet-simple':
    model = cst.ResNet(cst.ResidualBlock, [1, 1, 1, 1],num_classes=num_classes).to(device)
else:
    print('Unsupported model')
    
# Load model and move to device
model.load_state_dict(torch.load(os.path.join(os.getcwd(),'../Models' ,extractor_model_save_name), map_location=device))

model = nn.Sequential(*list(model.children())[:-1]) # Eliminate the last layer to use the model as a feature extractor 

model = cst.SiameseNetwork(model).to(device)

summary(model,((1,3,224,224),(1,3,224,224)))

# Define loss and optimizer

if loss_name=='contrastive':
    criterion = aux.ContrastiveLoss()
elif loss_name=='binary_cross_entropy':
    criterion = nn.BCELoss()
    
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=0.001)

# Define transformations (resize was arbitrary, normalize was requested from pytorch)
if use_data_augmentation==False:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
else:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Create custom dataset
dataset = aux.VerificationImageDataset(root_dir=root_dir, file_paths=file_paths_train, 
                                 classification_type=classification_type, 
                                 transform=transform, train=True, validation_split=0.25)

# Create training and validation subsets using the indices calculated during dataset.__init__()
train_subset = Subset(dataset, dataset.train_indices)
val_subset = Subset(dataset, dataset.val_indices)

# Create dataloaders for training and validation
train_loader = DataLoader(train_subset, batch_size=batch_size)
valid_loader = DataLoader(val_subset, batch_size=batch_size)

best_val_acc = 0  # Initialize best validation accuracy
epochs_without_improvement = 0  # Counter for epochs without improvement (early stopping)

# Training loop
for epoch in tqdm(range(num_epochs),leave=False):
    
    model.train()  # Set model to training mode
    running_loss = 0.0  # Set variables for evaluation
    correct_train = 0
    total_train = 0
    i = 0

    # Loads one batch at a time
    for images1, images2, labels in tqdm(train_loader, desc=f'Currently running epoch number {epoch+1}', leave=False):  
        # Move tensors to the configured device
        images1 = images1.to(device)
        images2 = images2.to(device)
        labels = labels.to(device)

        # Forward pass
        similarity = model(images1, images2)
        
        if loss_name=='binary_cross_entropy':
            labels = labels.float()
        
        loss = criterion(similarity.squeeze(), labels)
        
        #print(outputs)
        #print(criterion(outputs,labels))
        
        i += 1  # Counting batches
        running_loss += loss.item()  # Update running loss

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predicted = torch.reshape(similarity>classification_threshold,(1,labels.size(0)))
        total_train += similarity.size(0)
        
        #print(predicted)
        
        correct_train += (predicted == labels).sum().item()
        print(similarity)
        print(predicted)
        print(labels)

        del images1, images2, labels, similarity
        torch.cuda.empty_cache()
        gc.collect()
        
    # Validation loop, it doesn't influence training, it's just to keep track of the model overfitting or not
    model.eval()  # Set model to evaluation mode
    running_val_loss = 0.0  # Create variables for evaluation
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():   # Don't modify model during validation
        for images1, images2, labels in tqdm(valid_loader, leave=False):
            # Send tensors to device
            images1 = images1.to(device)
            images2 = images2.to(device)
            labels = labels.to(device)
            
            # Forward pass
            similarity = model(images1, images2)
            loss = criterion(similarity.squeeze(), labels)
            running_val_loss += loss.item()  # Update running validation loss

            predicted = torch.reshape(similarity>classification_threshold,(1,labels.size(0)))
            total_val += similarity.size(0)
            correct_val += (predicted == labels).sum().item()

            del images1, images2, labels, similarity

    # Calculate validation accuracy
    val_accuracy = 100 * correct_val / total_val

    # Print training and validation stats
    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {running_loss / i}, Training Accuracy: {100 * correct_train / total_train}%, '
        f'Validation Loss: {running_val_loss / len(valid_loader)}, Validation Accuracy: {val_accuracy}%')

    # Early stopping check
    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        epochs_without_improvement = 0  # Reset counter if there's improvement
        # Save the best model
        torch.save(model.state_dict(), os.path.join(os.getcwd(), model_save_name))
    else:
        epochs_without_improvement += 1

    # If patience is exceeded, stop training
    if epochs_without_improvement >= patience:
        print(f'Early stopping triggered after {patience} epochs without improvement in validation accuracy.')
        break

# Save the model (final model if no early stopping occurred)
torch.save(model.state_dict(), os.path.join(os.getcwd(),'../Models', model_save_name))