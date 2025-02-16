# PixelDigits Classification Using CNN Model
# Overview
* This repository involves training Convolutional Neural Network model using pytorch, which classifies handwritten pixel digits from 0 to 9 with the aiming of achieving accuracy score as high as possible. There's also an app deployment version of this project using HuggingFace platform via this URL: https://huggingface.co/spaces/PinKem/Handwritten_Digits_Classification
  
# Github Folder Explanation:
* data: contains MNIST dataset,includes raw data with 60.000 train images and 10.000 test images
* deploy_folder: contains necessary files which were used for deploying model
* going_modular: contains useful functional python files for reproductility
* models and checkpoint: model state storing
* notebook.ipynb: notebook file version used for implementing code

# Process
1. Setup dataset and exploring data
2. Create DataLoaders
3. Building a simple Baseline Model
4. Building a CNN model to extract feature
5. Import evaluation metrics
6. Create a train,test and compile function
7. Create a tracking plot (curve plot)
8. Compiling on different model
9. Testing CNN model on samples and real-world data
10. Combining steps and Write python file
11. Model Deployment

# Results
![image](https://github.com/user-attachments/assets/03a70e23-7bb9-4376-9ba4-e8396fe0343c)

![image](https://github.com/user-attachments/assets/4835b8f3-fc42-4595-a09d-b612ce38d38f)

    
# Techniques used
* Data augmentation
* EarlyStopping
* CustomDataset
* ...

# Requirement
* torch==1.12.0
* torchvision==0.13.0
* gradio==3.41.0
* httpx==0.28.1
* torchmetrics (latest version)
* torchinfo (latest version)
* ...





