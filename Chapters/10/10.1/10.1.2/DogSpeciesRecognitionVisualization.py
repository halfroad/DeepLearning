from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from glob import glob

# 120 species in all
classificationsNumber = 120

# Load the dataset

def Prepare(path):
    
    # Load the files via the method load_files() on sklearn
    # This method returns a object of dictionary comprised with releative path and its serial number
    data = load_files(path)
    
    # Convert the paths into Numpy object
    files = np.array(data["filenames"])
    # Sort the images by serial numbers
    originalTargets = np.array(data["target"])
    
    # Convert the file serial numbers into binary classified matrix (AKA one-hot encoding) via to_categorical()
    dogTargets = np_utils.to_categorical(originalTargets, classificationsNumber)
    
    # Return the paths of all the images, serial numbers and binary classfied matrix
    return files, originalTargets, dogTargets

# Load the dataset
files, originalTargets, dogTargets = Prepare("Images/")

# Load the list of dog species
# Glob is file operation related module. Return the path of file or directory via specific match pattern
# Here the operation is to return all the directories under Images
# Iterate the paths via list deduction, and truncate the string of species name of the dog
prefixLength = len("Images/n02085620-")
dogNames = [item[prefixLength: ] for item in sorted(glob("Images/*"))]

print("There are {} dog species in total.".format(len(dogNames)))
print("There are {} dog images in total.\n".format(len(files)))
