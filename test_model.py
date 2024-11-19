import torch
import os
from torchsummary import summary
import cv2
from torchvision import transforms
from PIL import Image

model_path=os.path.join(os.getcwd(),'CompCars/data/model.pt')
NUM_CLASSES=163

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

x=torch.nn.Sequential(model,torch.nn.Linear(in_features=1000,out_features=200))
x=torch.nn.Sequential(x,torch.nn.ReLU())
x=torch.nn.Sequential(x,torch.nn.Linear(in_features=200,out_features=NUM_CLASSES))
x=torch.nn.Sequential(x,torch.nn.Softmax(dim=1))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = x.to(device)

model.load_state_dict(torch.load(model_path,weights_only=True))
model.eval()

summary(model,(3,224,224))

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to a fixed size
    transforms.ToTensor(),          # Convert images to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])



path_to_image=os.path.join(os.getcwd(),'CompCars/data/cropped_image/64/1451/2012/0a928b2765d716.jpg')
label,*_=path_to_image[path_to_image.find('image')+6:].split('/')
print(label)

image=cv2.imread(path_to_image)
image = Image.fromarray(image.astype('uint8'), 'RGB')

image=transform(image)
image=image.cuda()

with torch.no_grad():
    out=model(image.unsqueeze(0))

print(out)
print(label)