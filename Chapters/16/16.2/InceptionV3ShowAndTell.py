import os
import json
import nltk
import shutil

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
    trainsJsonImagesProperties = trainsJsonDictionary["images"]
    validationsJsonImagesProperties = validationsJsonDictionary["images"]
    
    trainsJsonAnnotationsProperties = trainsJsonDictionary["annotations"]
    validationsJsonAnnotationsProperties = validationsJsonDictionary["annotations"]
    
    print("Trains: \r")
    print("Number of Images: {}, number of Image Properties: {}, number of Image Annotations: {}.".format(len(trainImagePaths), len(trainsJsonImagesProperties), len(trainsJsonAnnotationsProperties)))
    
    print("Validations: \r")
    print("Number of Images: {}, number of Image Properties: {}, number of Image Annotations: {}.".format(len(validationImagePaths), len(validationsJsonImagesProperties), len(validationsJsonAnnotationsProperties)))

def DownloadPunkt():
    
    # Download the module punkt and install
    nltk.download("punkt")
    
def RecordsStatistics():
    
    records = glob("../Exclusion/Datasets/Mscoco/*")
    
    print(records[: 10])
    
    trainRecords = []
    validationRecords = []
    testRecords = []
    
    for record in records:
        
        if record.startswith("../Exclusion/Datasets/Mscoco/train"):
            
            trainRecords.append(record)
            
        elif record.startswith("../Exclusion/Datasets/Mscoco/val"):
            
            validationRecords.append(record)
            
        elif record.startswith("../Exclusion/Datasets/Mscoco/test"):
            
            testRecords.append(record)
            
    print("Number of Train Records is {}".format(len(trainRecords)))
    print("Number of Validation Records is {}".format(len(validationRecords)))
    print("Number of Test Records is {}".format(len(testRecords)))
    
    
EnsureConsistency()

path = os.path.expanduser("~") + "/nltk_data/"

if not os.path.exists(path):
    shutil.copytree("../Exclusion/nltk_data/", os.path.expanduser("~") + "/nltk_data/")

DownloadPunkt()

RecordsStatistics()

if os.path.exists(path):
    shutil.rmtree(os.path.expanduser("~") + "/nltk_data")

'''

python3 data/build_mscoco_data.py --train_image_dir ../../Datasets/Trains/ --val_image_dir ../../Datasets/Validations/ --train_captions_file ../../Datasets/Merged/Trains.json --val_captions_file ../../Datasets/Merged/Validations.json --output_dir ../../Datasets/Mscoco/ --word_counts_output_file ../../Datasets/Mscoco/WordCounts.txt

'''
