# ISIC 2019: skin Lesion Analysis Towards Melanoma Detection

import pandas as pd
import os

# Move files
import shutil

def ExtractImages():
    
    root = "../Inventory/ISIC_2019/"
    groundTruthPath = "ISIC_2019_Training_GroundTruth.csv"
    
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
    
    trainingInput = "ISIC_2019_Training_Input/" * 2
    trainingDataset = "Train/Datasets/"
    
    melanomas = trainingGroundTruth[trainingGroundTruth["MEL"] == 1.0]
    
    for image in melanomas["image"]:
    
        source = root + trainingInput + image + ".jpg"
        destination = root + trainingDataset + "melanoma/" + image + ".jpg"
        
        shutil.copy(source, destination)
        
        print("File at {} copied to {}.".format(source, destination))
    
    
    melanocyticNevuses = trainingGroundTruth[trainingGroundTruth["NV"] == 1.0]
    
    for image in melanocyticNevuses["image"]:

        source = root + trainingInput + image + ".jpg"
        destination = root + trainingDataset + "melanocytic_nevus/" + image + ".jpg"
        
        shutil.copy(source, destination)
        
        print("File at {} copied to {}.".format(source, destination))
        
        
    basalCellCarcinomas = trainingGroundTruth[trainingGroundTruth["BCC"] == 1.0]
    
    for image in basalCellCarcinomas["image"]:

        source = root + trainingInput + image + ".jpg"
        destination = root + trainingDataset + "basal_cell_carcinoma/" + image + ".jpg"
        
        shutil.copy(source, destination)
        
        print("File at {} copied to {}.".format(source, destination))
    
    
    actinicKeratosises = trainingGroundTruth[trainingGroundTruth["AK"] == 1.0]
    
    for image in actinicKeratosises["image"]:

        source = root + trainingInput + image + ".jpg"
        destination = root + trainingDataset + "actinic_keratosis/" + image + ".jpg"
        
        shutil.copy(source, destination)
        
        print("File at {} copied to {}.".format(source, destination))
    
    
    benignKeratosisLikeLesions = trainingGroundTruth[trainingGroundTruth["BKL"] == 1.0]
    
    for image in benignKeratosisLikeLesions["image"]:

        source = root + trainingInput + image + ".jpg"
        destination = root + trainingDataset + "benign_keratosis-like_lesion/" + image + ".jpg"
        
        shutil.copy(source, destination)
        
        print("File at {} copied to {}.".format(source, destination))
        
    
    dermatofibromas = trainingGroundTruth[trainingGroundTruth["DF"] == 1.0]
    
    for image in dermatofibromas["image"]:

        source = root + trainingInput + image + ".jpg"
        destination = root + trainingDataset + "dermatofibroma/" + image + ".jpg"
        
        shutil.copy(source, destination)
        
        print("File at {} copied to {}.".format(source, destination))
        
    
    vascularLesions = trainingGroundTruth[trainingGroundTruth["VASC"] == 1.0]
    
    for image in vascularLesions["image"]:

        source = root + trainingInput + image + ".jpg"
        destination = root + trainingDataset + "vascular_lesion/" + image + ".jpg"
        
        shutil.copy(source, destination)
        
        print("File at {} copied to {}.".format(source, destination))
    
    squamousCellCarcinomas = trainingGroundTruth[trainingGroundTruth["SCC"] == 1.0]
    
    for image in squamousCellCarcinomas["image"]:

        source = root + trainingInput + image + ".jpg"
        destination = root + trainingDataset + "squamous_cell_carcinoma/" + image + ".jpg"
        
        shutil.copy(source, destination)
        
        print("File at {} copied to {}.".format(source, destination))
    
    unknowns = trainingGroundTruth[trainingGroundTruth["UNK"] == 1.0]
    
    for image in unknowns["image"]:

        source = root + trainingInput + image + ".jpg"
        destination = root + trainingDataset + "unknown/" + image + ".jpg"
        
        shutil.copy(source, destination)
        
        print("File at {} copied to {}.".format(source, destination))
    

# ExtractImages()
