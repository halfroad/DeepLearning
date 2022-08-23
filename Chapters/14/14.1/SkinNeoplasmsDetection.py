# ISIC 2019: skin Lesion Analysis Towards Melanoma Detection

import pandas as pd
import os

# Move files
import shutil

def Prepare():
    
    path = "../Inventory/ISIC_2019/ISIC_2019_Training_GroundTruth.csv"
    
    file = open(path)
    
    trainingGroundTruth = pd.read_csv(file)
    
    melanomas = trainingGroundTruth[trainingGroundTruth["MEL"] == 1.0]
    melanomas = trainingGroundTruth[trainingGroundTruth["NV"] == 1.0]
    melanomas = trainingGroundTruth[trainingGroundTruth["BCC"] == 1.0]
    melanomas = trainingGroundTruth[trainingGroundTruth["AK"] == 1.0]
    melanomas = trainingGroundTruth[trainingGroundTruth["BKL"] == 1.0]
    melanomas = trainingGroundTruth[trainingGroundTruth["DF"] == 1.0]
    melanomas = trainingGroundTruth[trainingGroundTruth["VASC"] == 1.0]
    melanomas = trainingGroundTruth[trainingGroundTruth["SCC"] == 1.0]
    melanomas = trainingGroundTruth[trainingGroundTruth["UNK"] == 1.0]
    
    print(melanomas)
    

Prepare()