import os
import random
import numpy as np
from collections import defaultdict

"""
The authors took a subset of the whole dataset and split it into training and testing
then created a train_test_split folder, with a classification folder inside
and inside that two txt files, one with the list of the paths to the training images,
one with the paths to the test images.

This code is used to mimick this.
"""

# Paths and split percentages
root_path = os.path.join(os.getcwd(),
                          '../CompCars/data/cropped_image')     # Root path of where the images are
write_path = os.path.join(os.getcwd(),
                           '../CompCars/data/train_test_split3/classification')     # Path of where to write the txt files with the train test splits
percentage_to_use = 1.0     # Percentage of the whole data to use
percentage_to_test = 0.2    # Percentage of the used data to be used during test only
np.random.seed(0)

# Function to extract labels from paths, set the index to return as 0 for make label, 1 for model label
def extract_class_label(file_path):
    return file_path.split('/')[1]

# Walk through the folder tree to extract all labels and make the splits evenly
files_by_class = defaultdict(list)
for dirpath, dirnames, filenames in os.walk(root_path):
    for file in filenames:
        rel_path = os.path.join(dirpath, file).split(root_path + '/')[1]    # Eliminate the whole part before the end of root_path
        class_label = extract_class_label(rel_path)     
        files_by_class[class_label].append(rel_path)    # Save the path into the corresponding slot for the label

# Print the total number of samples per class to check
print("Total samples per label:")
for class_label, file_list in files_by_class.items():
    print(f"Label {class_label}: {len(file_list)} samples")

# Perform even split by label
train_files = []
test_files = []
for class_label, file_list in files_by_class.items():
    file_list = np.array(file_list)
    num_files = len(file_list)
    num_to_use = int(percentage_to_use * num_files)
    num_to_test = int(num_to_use * percentage_to_test)
    
    # Shuffle indices and split
    indices = np.arange(num_files)
    np.random.shuffle(indices)
    test_indices = indices[:num_to_test]
    train_indices = indices[num_to_test:num_to_use]
    
    train_files.extend(file_list[train_indices])
    test_files.extend(file_list[test_indices])

# Save train and test splits
os.makedirs(write_path, exist_ok=True)

with open(os.path.join(write_path, 'train.txt'), 'w') as file:
    for path in train_files:
        file.write(path + '\n')

with open(os.path.join(write_path, 'test.txt'), 'w') as file:
    for path in test_files:
        file.write(path + '\n')

# Print the number of samples per label in train and test sets to check
train_counts = defaultdict(int)
test_counts = defaultdict(int)

for path in train_files:
    label = extract_class_label(path)
    train_counts[label] += 1

for path in test_files:
    label = extract_class_label(path)
    test_counts[label] += 1

print("\nSamples per label in training set:")
for label, count in train_counts.items():
    print(f"Label {label}: {count} samples")

print("\nSamples per label in testing set:")
for label, count in test_counts.items():
    print(f"Label {label}: {count} samples")

# Print dimensions
print(f"Training files: {len(train_files)}, Testing files: {len(test_files)}")
