import numpy as np
import os
from tensorflow.keras.models import load_model
from PIL import ImageOps
from tensorflow.keras.preprocessing import image# used for preproccesing 
model = load_model('flowers.h5')
print("Loaded model from disk")
classs = { 1:"Daisy",
           2:"Dandelion",
           3:"Rose",
           4:"Sunflower",
           5:"Tulip"
           }
Img=30
def classify(img_file):
    test_image=image.load_img(img_file)
    test_image=ImageOps.grayscale(test_image)
    test_image = test_image.resize((30,30))
    test_image = np.expand_dims(test_image, axis=0)
    test = np.array(test_image).reshape(-1,Img,Img,1)
    result = model.predict_classes(test)[0]
    sign = classs[result + 1]
    print(sign)
    
print("Obtaining Images & its Labels..............")
path='D:/python/dl programs/Flowers Recognition/data/test'
files=[]
print("Dataset Loaded")
# r=root,d=directories,f=files
for r,d,f in os.walk(path):
    for file in f:
        if '.jpeg' or '.jpg' or '.png' or '.JPG' in file:
            files.append(os.path.join(r,file))
for f in files:
    classify(f)
    print('\n')
