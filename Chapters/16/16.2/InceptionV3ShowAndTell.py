import os
import json

from glob import glob

def EnsureConsistency():
    
    curentDirectory = os.getcwd()
    
    trainImagesPath = curentDirectory + "/../Exclusion/Datasets/Trains/"
    trainsJsonPath = curentDirectory + "/../Exclusion/Datasets/Merged/Trains.json"
    
    validationIamgesPath = curentDirectory + "/../Exclusion/Datasets/Validations/"
    validationsJsonPath = curentDirectory + "/../Exclusion/Datasets/Merged/Validations.json"
    
    # Acquire all the paths of Train images
    trainImagePaths = glob(trainImagesPath + "*")
    
    # Acquire all the paths of Validation images
    validationImagePaths = glob(validationIamgesPath + "*")
    
    # Load the annotations for Trains
    with open(trainsJsonPath, encoding="utf-8") as f:
        
        trainsJsonDictionary = json.load(f)
        
    # Load the annotations for Validations
    with open(validationsJsonPath, encoding="utf-8") as f:
        
        validationsJsonDictionary = json.load(f)
        
    # Print basic keys
    print("Keys in trainsJsonDictionary: {}".format(trainsJsonDictionary.keys()))
    print("Keys in validationsJsonDictionary: {}".format(validationsJsonDictionary.keys()))
    
    # Statistics of trains set and validations set
    trainsJsonImagesPaths = trainsJsonDictionary["images"]
    validationsJsonImagesPaths = validationsJsonDictionary["images"]
    
    trainsJsonAnnotationsPaths = trainsJsonDictionary["annotations"]
    validationsJsonAnnotationsPaths = validationsJsonDictionary["annotations"]
    
    print("Trains: \r")
    print("Number of Images: {}, number of Image Properties: {}, number of Image Annotations: {}.".format(len(trainImagePaths), len(trainsJsonImagesPaths), len(trainsJsonAnnotationsPaths)))
    
    print("Validations: \r")
    print("Number of Images: {}, number of Image Properties: {}, number of Image Annotations: {}.".format(len(validationImagePaths), len(validationsJsonImagesPaths), len(validationsJsonAnnotationsPaths)))

EnsureConsistency()