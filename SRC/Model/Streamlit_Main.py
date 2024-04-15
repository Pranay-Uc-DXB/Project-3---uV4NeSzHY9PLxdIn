# %%
import streamlit as st
from PIL import Image
from util import classify, set_background

import torch
import os 

# Importing all the packages

from torchvision.transforms import ToTensor
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import RandomHorizontalFlip
from torchvision.transforms import RandomRotation
from torchvision.transforms import Normalize
from torchvision.transforms import Resize
from torchvision.transforms import transforms


from torchvision import transforms
from torchvision.transforms import Normalize
import numpy as np

# %%
set_background('C:/Users/97158/Desktop/Apziva/Project 4-MonReader/Model and other images/Screenshot 2024-04-04 123307.png')


# %%
#st.title('Page Flipping Classification')
st.write('<h1 style="color:black;">Page Flipping Classification</h1>', unsafe_allow_html=True)

# %%
st.header(':blue[Please upload an open book image]')

# %%
file= st.file_uploader('',type=['jpeg','jpg','png'])

# %%
with open('C:/Users/97158/Desktop/Apziva/Project 4-MonReader/labels.txt','r') as f:
    class_names=[line.strip().split(' ')[1] for line in f.readlines()]
    f.close()

# %%
import torch

Streamlit_Model_path= os.path.join("model_output","Strmlt_model.pth")


# Load the saved model
model = torch.load(Streamlit_Model_path)

# %%

Mean= [0.485, 0.456, 0.406]
STD= [0.229, 0.224, 0.225]
Image_size=768

Transform=transforms.Compose([
    Resize((Image_size, Image_size)),
    #RandomResizedCrop(Image_size),
    #RandomHorizontalFlip(),
    #RandomRotation(90),
    ToTensor(),
    Normalize(Mean, STD)])

# display image
if file is not None:
    image=Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    image = Transform(image)


    #classify image
    class_name, confidence_score = classify(image, model, class_names)

    # write classification
    st.write('# :gray[  Prediction: {}]'.format(class_name))
    
    
    st.write('## :gray[  Score: {}]'.format(np.round(confidence_score,3)))
