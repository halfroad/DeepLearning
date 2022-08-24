# ISIC 2019: skin Lesion Analysis Towards Melanoma Detection

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from matplotlib import image


# Move files
import shutil


root = "../Inventory/ISIC_2019/"
trainingDatasets = "Train/Datasets/"
groundTruthPath = "ISIC_2019_Training_GroundTruth.csv"
trainingInput = "ISIC_2019_Training_Input/" * 2

def ExtractImages():
    
    file = open(root + groundTruthPath)
    
    trainingGroundTruth = pd.read_csv(file)
    
    '''
    
    MEL   melanoma
    NV    melanocytic nevus
    BCC   basal cell carcinoma
    AK    actinic keratosis
    BKL   benign keratosis-like lesion
    DF    dermatofibroma
    VASC  vascular lesion
    SCC   squamous cell carcinoma
    UNK

    '''
    
    melanomas = trainingGroundTruth[trainingGroundTruth["MEL"] == 1.0]
    
    for image in melanomas["image"]:
    
        source = root + trainingInput + image + ".jpg"
        destination = root + trainingDatasets + "melanoma/" + image + ".jpg"
        
        shutil.copy(source, destination)
        
        print("File at {} copied to {}.".format(source, destination))
    
    
    melanocyticNevuses = trainingGroundTruth[trainingGroundTruth["NV"] == 1.0]
    
    for image in melanocyticNevuses["image"]:

        source = root + trainingInput + image + ".jpg"
        destination = root + trainingDatasets + "melanocytic_nevus/" + image + ".jpg"
        
        shutil.copy(source, destination)
        
        print("File at {} copied to {}.".format(source, destination))
        
        
    basalCellCarcinomas = trainingGroundTruth[trainingGroundTruth["BCC"] == 1.0]
    
    for image in basalCellCarcinomas["image"]:

        source = root + trainingInput + image + ".jpg"
        destination = root + trainingDatasets + "basal_cell_carcinoma/" + image + ".jpg"
        
        shutil.copy(source, destination)
        
        print("File at {} copied to {}.".format(source, destination))
    
    
    actinicKeratosises = trainingGroundTruth[trainingGroundTruth["AK"] == 1.0]
    
    for image in actinicKeratosises["image"]:

        source = root + trainingInput + image + ".jpg"
        destination = root + trainingDatasets + "actinic_keratosis/" + image + ".jpg"
        
        shutil.copy(source, destination)
        
        print("File at {} copied to {}.".format(source, destination))
    
    
    benignKeratosisLikeLesions = trainingGroundTruth[trainingGroundTruth["BKL"] == 1.0]
    
    for image in benignKeratosisLikeLesions["image"]:

        source = root + trainingInput + image + ".jpg"
        destination = root + trainingDatasets + "benign_keratosis-like_lesion/" + image + ".jpg"
        
        shutil.copy(source, destination)
        
        print("File at {} copied to {}.".format(source, destination))
        
    
    dermatofibromas = trainingGroundTruth[trainingGroundTruth["DF"] == 1.0]
    
    for image in dermatofibromas["image"]:

        source = root + trainingInput + image + ".jpg"
        destination = root + trainingDatasets + "dermatofibroma/" + image + ".jpg"
        
        shutil.copy(source, destination)
        
        print("File at {} copied to {}.".format(source, destination))
        
    
    vascularLesions = trainingGroundTruth[trainingGroundTruth["VASC"] == 1.0]
    
    for image in vascularLesions["image"]:

        source = root + trainingInput + image + ".jpg"
        destination = root + trainingDatasets + "vascular_lesion/" + image + ".jpg"
        
        shutil.copy(source, destination)
        
        print("File at {} copied to {}.".format(source, destination))
    
    squamousCellCarcinomas = trainingGroundTruth[trainingGroundTruth["SCC"] == 1.0]
    
    for image in squamousCellCarcinomas["image"]:

        source = root + trainingInput + image + ".jpg"
        destination = root + trainingDatasets + "squamous_cell_carcinoma/" + image + ".jpg"
        
        shutil.copy(source, destination)
        
        print("File at {} copied to {}.".format(source, destination))
    
    unknowns = trainingGroundTruth[trainingGroundTruth["UNK"] == 1.0]
    
    for image in unknowns["image"]:

        source = root + trainingInput + image + ".jpg"
        destination = root + trainingDatasets + "unknown/" + image + ".jpg"
        
        shutil.copy(source, destination)
        
        print("File at {} copied to {}.".format(source, destination))
    

def Prepare():
        
    file = open(root + groundTruthPath)
    
    trainingGroundTruth = pd.read_csv(file)
    
    print("trainingGroundTruth.shape = {}".format(trainingGroundTruth.shape))
    
    return trainingGroundTruth
    
def Visualize(trainingGroundTruth):
    
    # Use the default style for matplotlib when drawing
    plt.style.use("default")
    
    # Create 9 drawing objects, 3 rows and 3 columns
    figure, axes = plt.subplots(nrows = 3, ncols = 3)
    
    # Set the size of figure
    figure.set_size_inches(10, 9)
    
    # Select 9 numbers randomly, means 9 disease images
    randomNumbers = np.random.choice(len(trainingGroundTruth), 9)
    
    print(type(trainingGroundTruth))
    
    # Select 9 images from dataset, and its pathes
    randomGroundTruths = trainingGroundTruth.iloc[randomNumbers]
        
    # Extract the skin cancer names from the paths of 9 images
    names = root + trainingInput + randomGroundTruths["image"] + ".jpg"
    columns = randomGroundTruths.columns[randomGroundTruths.eq(1.0).all()]
    
    print(columns)
    
    index = 0
    
    # Row
    for row in range(3):
        
        for column in range(3):
            
            # Read the value from image
            _image = image.imread(names[index])
            
            # Grab the Axes object according to [row, column]
            ax = axes[row, column]
            
            ax.imshow(_image)
            
            # Set the title with the name of skin cancer
            ax.set_xlable(names[index])
            
    plt.show()
    
# ExtractImages()
trainingGroundTruth = Prepare()
Visualize(trainingGroundTruth)
