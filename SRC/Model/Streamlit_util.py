import base64

import streamlit as st
from PIL import ImageOps, Image
import numpy as np

import streamlit as st
from PIL import Image

import torch
import os 

# Importing all the packages

import torch
from torch import nn
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import RandomHorizontalFlip
from torchvision.transforms import RandomRotation
from torchvision.transforms import Normalize
from torchvision.transforms import Resize
from torchvision.transforms import transforms
from torch.nn import CrossEntropyLoss
from torch.nn import Softmax
from torch.nn import ReLU
from torch.nn import Dropout
from torch import optim
from tqdm import tqdm
from matplotlib import pyplot as plt 
import argparse

from torchvision import transforms
from torchvision.transforms import Normalize
import pandas as pd

import sys
sys.argv=['']
del sys



ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default="mobilenet", choices=["vgg", "resnet","mobilenet"], help="name of the backbone model")
args = vars(ap.parse_args())



# check if the name of the backbone model is VGG
if args["model"] == "vgg":
	# load VGG-11 model
	baseModel = torch.hub.load("pytorch/vision:v0.10.0", "vgg11",weights='VGG11_Weights.DEFAULT', skip_validation=True)
# otherwise, the backbone model we will be using is a ResNet
elif args["model"] == "resnet":
	# load ResNet 18 model
	baseModel = torch.hub.load("pytorch/vision:v0.10.0", "resnet18",weights='ResNet18_Weights.DEFAULT', skip_validation=True)
elif args['model']=='mobilenet':
    baseModel=torch.hub.load('pytorch/vision:v0.10.0','mobilenet_v2',weights='MobileNet_V2_Weights.DEFAULT',skip_validation=True) 	


from torch.nn import Linear, Module, Sequential, Conv2d


class Classifier(Module):
    def __init__(self, baseModel, numclasses, model):
        super().__init__()
        self.baseModel = baseModel


        if model == 'vgg':
            for module, param in zip(baseModel.modules(),baseModel.parameters()):
                 if isinstance(module, nn.BatchNorm2d):
                      param.requires_grad=False

            # Extracting the number of input features from the last layer of VGG
            num_features = baseModel.classifier[6].out_features
            
            # Defining the extra layer
            additional_layers=[Linear(num_features,512),
                               ReLU(inplace=True),
                               Dropout(0.2),
                               Linear(512,256),
                               ReLU(inplace=True),
                               Dropout(0.2),
                               Linear(256,numclasses)]
            self.baseModel.fc=Sequential(*additional_layers)
            
        elif model=='resnet':
            for module, param in zip(baseModel.modules(),baseModel.parameters()):
                if isinstance(module, nn.BatchNorm2d):
                    param.requires_grad=False
                       
            num_features=baseModel.fc.out_features

            additional_layers=[Linear(num_features,512,bias=True),
                               ReLU(inplace=True),
                               Dropout(0.2),
                               Linear(512,256,bias=True),
                               ReLU(inplace=True),
                               Dropout(0.2),
                               Linear(256,numclasses,bias=True)]
                               
            self.baseModel.classifier=Sequential(*additional_layers)
                   

        elif model=='mobilenet':       
            for module, param in zip(baseModel.modules(),baseModel.parameters()):
                 if isinstance(module, nn.BatchNorm2d):
                      param.requires_grad=False
             
            num_features=baseModel.classifier[-1].out_features
            additional_layers=[Dropout(0.2),
                                Linear(num_features,512,bias=True),
                                ReLU(inplace=True),
                                Dropout(0.2),
                                Linear(512,256,bias=True),
                                ReLU(inplace=True),
                                Dropout(0.2),
                                Linear(256,numclasses,bias=True)]
            baseModel.fc=Sequential(*additional_layers)
              
        

    def forward(self, x):
        features = self.baseModel(x)
        
        # Applying the extra layer and ReLU activation
        if 'vgg' in self.baseModel.__class__.__name__.lower():
              logits = self.baseModel.fc(features)
              
        elif 'resnet' in self.baseModel.__class__.__name__.lower():
              logits = self.baseModel.classifier(features)  

        elif 'mobilenet' in self.baseModel.__class__.__name__.lower():        
              logits =self.baseModel.fc(features)

        return logits        


# Creating the directory if it doesn't exist
os.makedirs("model_output", exist_ok=True)

# Defining paths to store trained model
VGG_Model_path= os.path.join("model_output","VGG_model.pth")
RESNET_Model_path= os.path.join("model_output","ResNet_model.pth")
MobileNet_Model_path= os.path.join("model_output","MobileNet_model.pth")
Streamlit_Model_path= os.path.join("model_output","Strmlt_model.pth")

#Loading the model state and inititalizing the loss function
# build the custom model
model = Classifier(baseModel=baseModel,numclasses=2, model=args['model'])


if args['model']=='vgg':
    model.load_state_dict(torch.load(VGG_Model_path))
elif args['model']=='resnet':
    model.load_state_dict(torch.load(RESNET_Model_path))
elif args['model']=='mobilenet':
    model.load_state_dict(torch.load(MobileNet_Model_path))  


torch.save(model, Streamlit_Model_path)





def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.

    Parameters:
        image_file (str): The path to the image file to be used as the background.

    Returns:
        None
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)



def classify(image, model, class_names):
        
    soft=Softmax()


        # Putting the model in evaluation mode with gradients turned off
    with torch.no_grad():
        model.eval() 

        test_pred=[]
        
        
        # Making the predictions and calculating the validation loss
            
        logit=model(image.unsqueeze(0))

        # Getting predictions and calculating all the correct predictions
        pred=soft(logit)
        test_pred.extend(pred.argmax(axis=1).detach().cpu().numpy())

        test_pred=np.array(test_pred)
        index = 0 if test_pred== 0 else 1
        class_name = class_names[index]
        #test_targets=np.array(testDataset.targets)
        #confidence_score=classification_report(test_targets,test_pred,target_names=testDataset.classes))
        confidence_score = pred[0][index]
        return class_name, confidence_score

