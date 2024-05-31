# Computer Vision for Everyone


# Introduction
Worked on a new mobile document digitization experience for the blind, researchers, and everyone else in need for fully automatic, highly fast and quality-driven document scanning services. This service is going to be packed in a mobile app and all the user needs to do is flip pages; rest everything is handled by the app. 

So, how does the app work? 

The app will detect page flips from low-resolution camera preview and take a high-resolution picture of the document, recognizing its corners and cropping it accordingly. It will then dewarp the cropped document to obtain a bird's eye view, sharpen the contrast between the text and the background and finally recognize the text with formatting kept intact. The formatting is further corrected by app’s ML powered reactor.


# Methodology:

In this project, I developed and tested 4 different CNNs in Pytorch, namely:
1)  LeNet-5
2)  VGG16
3)  ResNet18
4)  MobileNetV2.

The aim was to pack the service app for mobile compatibility with total size less than 40MB and a minimum reporting F1 score of >=0.90.

Below, I demonstrate the final product that uses MobileNet V2 architecture in synergy with a custom classifier head (implementing transfer learning). Enjoy!

https://github.com/Pranay-Uc-DXB/Project-3-uV4NeSzHY9PLxdIn/assets/62109186/5ef9b844-871f-4239-98fc-642340fb98ac


# Conclusion:

<img width="449" alt="image" src="https://github.com/Pranay-Uc-DXB/Project-3-uV4NeSzHY9PLxdIn/assets/62109186/49be66f4-f52a-496d-897b-a300695109da"> <img width="194" alt="image" src="https://github.com/Pranay-Uc-DXB/Project-3-uV4NeSzHY9PLxdIn/assets/62109186/79602059-addc-475e-b2f5-ab0348d1a540">

When using transfer learning, despite subjecting training data to various data augmentation techniques, my model's performance (training accuracy) did not substantially improve. This indicated that data augmentation and transfer learning only works on specific use cases depending on the application. My testing accuracy was far better than my training accuracy, indicating underfitting, However, this may be the case due to the measures taken to reduce overfitting. Some of the measures included regularization (including dropouts), and introducing double non-linearity. Nevertheless, upon deploying the model, the classification model performed very well with only 5 false negatives.  

Transfer learning may not always be the best approach towards computer vision projects. In my case, a custom LetNet architecture w/o transfer learning outperformed other networks while training and testing. 

My Results with a custom classifier mounted on MobileNet V2 Architecture:
1) Total size/weight of the model obtained: 16.4MB (Target: <20MB)
2) Model F1 Score: 0.99 (Target: >=0.90)
