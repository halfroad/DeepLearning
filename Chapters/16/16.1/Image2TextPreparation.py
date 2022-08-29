# import tensorflow as tf
# print(tf.test.gpu_device_name())

import json
import matplotlib.pyplot as plt
import matplotlib.image as image
import os
import shutil

from glob import glob
from tqdm import tqdm

def Parse(path):
    
    with open(path, encoding="utf-8") as f:
        
        dictionary = json.load(f)
        
        print(dictionary.keys())
        
        images = dictionary["images"]
        annotations = dictionary["annotations"]
        
        print("images.length = {}, annotations.length = {}".format(len(images), len(annotations)))
        
        print(dictionary["images"][: 3])
        print(dictionary["annotations"][: 3])
        print(dictionary["info"])
        print(dictionary["licenses"])
        
        return dictionary, images, annotations
        
def Preview(name, annotations):
    
    '''

    Preview the image relies on its path
    
    '''
    
    # Iterate the annotations array
    
    for annotation in annotations:
        
        # Each annotation has a key of image_id, use this key to check whether the same with a specified image
        # If the same image_id, that means the annotation of an image is discovered
        if annotation["image_id"] == name:
            
            # Iterate the images array
            for prop in images:
                
                # Remove the extension name
                fileName = prop["file_name"][: -4]
                # If the name of image matches the name of spcified image, that can be reckoned the image is discovered
                
                if int(fileName) == name:
                    
                    # Read the image discovered
                    _image = image.imread(os.getcwd() + "/../Exclusion/val2017/" + prop["file_name"])
                    
                    # Hide the grid
                    
                    plt.grid(False)
                    
                    # Show the image
                    plt.imshow(_image)
                    
                    print("The properties are:")
                    print(prop)
                    
                    break
                
            print("The annotations of image are:")
            print(annotation)
            
            break
        
    plt.show()
    
def Split():
    
    # Read aall the paths of image under val2017, * means the wildcard
    val2017 = os.getcwd() + "/../Exclusion/val2017/*"
    
    paths = glob(val2017)
    
    # Set the path of train and validation.
    trainImagesPath = os.getcwd() + "/../Exclusion/Datasets/Trains/"
    trainImagesPaths = glob(trainImagesPath + "*")
    
    if len(trainImagesPaths) == 0:
        
        print("Copy images into Trains")
        
        # Split 4500 images for train set
        for path in tqdm(paths[: 4500]):
            
            fileName = os.path.basename(path)
            destination = trainImagesPath + fileName
            
            shutil.copy(path, destination)
                        
        trainImagesPath = glob(trainImagesPath + "*")
    
    validationImagesPath = os.getcwd() + "/../Exclusion/Datasets/Validations/"
    validationImagesPaths = glob(validationImagesPath + "*")
    
    if len(validationImagesPaths) == 0:
        
        print("Copy images into Vaidations")
        
        # Split 500 images for validation set
        for path in tqdm(paths[4500:]):
            
            fileName = os.path.basename(path)
            destination = validationImagesPath + fileName
            
            shutil.copy(path, destination)
                        
        validationImagesPaths = glob(validationImagesPath + "*")
        
    print(validationImagesPaths[: 10])
        
    return trainImagesPaths, validationImagesPaths
    
def LinkImagesAnnotations(paths, images, annotations, isTrain = False):
    
    # Parental folder
    folder = len(os.getcwd() + "/../Exclusion/Datasets/" + ("Trains/" if isTrain else "Validations/"))
    
    # Name of image (Without extension name)
    length = len("000000570664")
    
    imagesJson = []
    annotationsJson = []
    
    for path in tqdm(paths):
        
        # Find the corresponding properties of train images, and append to array
        fileName = path[folder: folder + length]
        
        for annotation in annotations:
            
            if annotation["image_id"] == int(fileName):
                
                annotationsJson.append(annotation)
                
                break
            
        fileName = path[folder:]
        
        for prop in images:
            
            if prop["file_name"] == fileName:
                
                annotationsJson.append(prop)
                
                break
            
    return imagesJson, annotationsJson

def Merge(dictionary, imagesJson, annotationsJson, storagePath):
    
    # Read the info property
    information = dictionary["info"]
    
    # Read the licenses property
    licenses = dictionary["licenses"]
    
    # Construct the json
    jsonObject = {"info": information, "licenses": licenses, "images": imagesJson, "annotations": annotationsJson}
    
    # Store the json
    with open(storagePath, "w") as f:
        
        f.write(json.dumps(jsonObject))

dictionary, images, annotations = Parse(os.getcwd() + "/../Exclusion/annotations/captions_val2017.json")

Preview(int("000000037777"), annotations)
Preview(int("000000005037"), annotations)

trainPaths, validationPaths = Split()

mergedTrainsPath = os.getcwd() + "/../Exclusion/Datasets/Merged/Trains.json"

if not os.path.exists(mergedTrainsPath):
    
    print("Linking images and annotations for Trains Set")
    
    trainImagesJson, trainAnnotationsJson = LinkImagesAnnotations(trainPaths, images, annotations, isTrain = True)

    print("Merging images and annotations for Trains Set")
    
    # Merge the trains set
    Merge(dictionary, trainImagesJson, trainAnnotationsJson, mergedTrainsPath)

mergedValidationsPath = os.getcwd() + "/../Exclusion/Datasets/Merged/Validations.json"

if not os.path.exists(mergedValidationsPath):
    
    print("Linking images and annotations for Validations Set")

    validationImagesJson, validationAnnotationsJson = LinkImagesAnnotations(validationPaths, images, annotations)

    print("Merging images and annotations for Validations Set")
    
    # Merge the vaidations set
    Merge(dictionary, validationImagesJson, validationAnnotationsJson, mergedValidationsPath)

print("All Ready")