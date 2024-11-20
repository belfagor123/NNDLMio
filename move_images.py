import os
import shutil
from torchvision import transforms, datasets
import numpy as np
from torch.utils.data import SubsetRandomSampler

root_path = os.path.join(os.getcwd(), '../CompCars/data/cropped_image')
destination_path = os.path.join(os.getcwd(), '../CompCars/data/split_cropped_image')

test_perc = 0.3

# Already normalize the images so that you don't have to do it during training or testing
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize,
])

# Load the whole dataset with from the folder structure
full_dataset = datasets.ImageFolder(
    root=root_path, transform=transform
)

# Extract some samples to be only used during test 
num_full = len(full_dataset)
indices = list(range(num_full))
split = int(np.floor(test_perc * num_full))
np.random.seed(42)  # Reproducibility, can be changed
np.random.shuffle(indices)
train_idx, test_idx = indices[split:], indices[:split]

# Function to copy images while preserving folder structure
def copy_images(indices, dataset, destination_dir):
    for idx in indices:
        img_path, _ = dataset.imgs[idx]
        
        # Get the relative path to preserve folder structure
        relative_path = os.path.relpath(img_path, start=root_path)
        
        # Define new path
        dest_class_dir = os.path.join(destination_dir, os.path.dirname(relative_path))
        
        # Make sure directories exist
        os.makedirs(dest_class_dir, exist_ok=True)
        
        # Copy image
        shutil.copy(img_path, dest_class_dir)

# Copy images to the train and test directories
train_path = os.path.join(destination_path, 'train')
test_path = os.path.join(destination_path, 'test')

copy_images(train_idx, full_dataset, train_path)
copy_images(test_idx, full_dataset, test_path)

