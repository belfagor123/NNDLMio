import os
import cv2
from tqdm import tqdm

root_path=os.path.join(os.getcwd(),'../CompCars/data')

for subdir, dirs, files in tqdm(os.walk(os.path.join(root_path,'image'))):

    for file in files:

        image_path=os.path.join(subdir,file)
        info_path=image_path.replace('image','label').replace('jpg','txt')

        with open(info_path,'r') as f:

            x1,y1,x2,y2=[int(val) for val in f.readlines()[2].replace('\n','').split(sep=' ')]
            image = cv2.imread(image_path)
            cropped_image = image[y1:y2, x1:x2, :]

            make, model, year, filename = image_path[image_path.find('image')+6:].split('/')

            cropped_path=os.path.join(root_path, 'cropped_image', make, model, year)

            if not os.path.exists(cropped_path):
                try:
                    os.makedirs(cropped_path)
                except:
                    pass
            if not os.path.exists(os.path.join(cropped_path,filename)):
                try:
                    cv2.imwrite(os.path.join(cropped_path, filename), cropped_image)
                except:
                    pass

