import os
import matplotlib.pyplot as plt

path = os.path.join(os.getcwd(),
                           '../CompCars/data/splits/train_test_split_part_model_100_80_20_0/classification')

label_type='model'
train_path = os.path.join(path,'train.txt')
test_path = os.path.join(path,'test.txt')

if label_type=='make':
    id_label=0
elif label_type=='model':
    id_label=1

train_data={}
test_data={}

with open(train_path,'r') as file:
    for line in file:
        label=int(line.split('/')[id_label])
        if label in train_data:
            train_data[label]+=1
        else:
            train_data[label]=1

with open(test_path,'r') as file:
    for line in file:
        label=int(line.split('/')[id_label])
        if label in test_data:
            test_data[label]+=1
        else:
            test_data[label]=1

tot_train=sum(train_data.values())
tot_test=sum(test_data.values())

plt.bar(train_data.keys(),[val/tot_train*100 for val in train_data.values()],width=1.5,label='Train')
plt.bar(test_data.keys(),[val/tot_test*100 for val in test_data.values()],width=0.5,label='Test')
plt.xlabel('Car make number')
plt.ylabel('Percentage of samples in the class (%)')

plt.legend()

plt.show()


