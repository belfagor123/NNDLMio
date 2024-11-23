import os

# Function to parse the dataset
def count_car_makes_and_models(file_paths):
    car_makes = set()
    car_models = set()

    for file_path in file_paths:
        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"File {file_path} not found.")
            continue

        with open(file_path, 'r') as file:
            for line in file:
                # Extract make and model from the path
                path_parts = line.strip().split('/')  # Assuming path is separated by '/'
                if len(path_parts) >= 2:  # Ensure the path has at least make and model
                    car_makes.add(path_parts[0])  # car_make_id
                    car_models.add(f"{path_parts[0]}/{path_parts[1]}")  # car_make_id/car_model_id
    
    return len(car_makes), len(car_models)

# Paths to your training and testing text files
training_file = os.path.join(os.getcwd(),'../CompCars/data/train_test_split/classification/train.txt')
testing_file = training_file.replace('train','test')

# Count car makes and models
file_paths = [training_file, testing_file]
num_car_makes, num_car_models = count_car_makes_and_models(file_paths)

print(f"Number of unique car makes: {num_car_makes}")
print(f"Number of unique car models: {num_car_models}")
