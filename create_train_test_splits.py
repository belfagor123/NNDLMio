import os
import random as random
import numpy as np

root_path=os.path.join(os.getcwd(),'../CompCars/data/cropped_image')
write_path=os.path.join(os.getcwd(),'../CompCars/data/train_test_split2/classification')
percentage_to_use=1.0
percentage_to_test=0.2

files=[]
os.makedirs(write_path, exist_ok=True)

for dirpath,dirnames,filenames in os.walk(root_path):
    for file in filenames:    
        files.append(os.path.join(dirpath,file).split(root_path+'/')[-1])
        
files=np.array(files)
total_files=len(files)
num_to_use=int(percentage_to_use*total_files)
ind_to_use=np.array(random.sample(range(total_files),num_to_use))
num_to_test=int(num_to_use*percentage_to_test)
ind_to_test=ind_to_use[:num_to_test]
ind_to_train=ind_to_use[num_to_test:]

files_train=files[ind_to_train]
files_test=files[ind_to_test]

with open(os.path.join(write_path,'train.txt'),'a') as file:
    for path in files_train:
        file.write(path + '\n')
        
with open(os.path.join(write_path,'test.txt'),'a') as file:
    for path in files_test:
        file.write(path + '\n')