import keras
from tqdm import tqdm
import numpy as np
from PIL import ImageFile

import sys
sys.path.insert(1, "../10.1.2/")

from DogSpeciesRecognitionVisualization import Prepare, Split

def Path2Tensor(imagePath):
    
    # Load the image
    # Utilizing the PIL library to load image object. The load_image() method will return a PIL object
    image = keras.utils.load_img(imagePath, target_size = (224, 224, 3))
    # Convert the PIL object to 3 dimensions vector(224, 224, 3)
    x = keras.utils.img_to_array(image)
    
    # Convert the 3 dimensions vector to 4 dimensions vector (1, 224, 224, 3)
    return np.expand_dims(x, axis = 0)

def Paths2Tensor(imagePaths):
    
    # Module tqdm will show a progress bar by passing an array of all images
    # Convert all the images to the vector of numpy values, and return the array
    tensors = [Path2Tensor(path) for path in tqdm(imagePaths)]
    
    return np.vstack(tensors)

def CreateTensors(X_train, X_validation, X_test):
    
    # Set the Truncated Images to true to avoid the IO error when reading image by PIL
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    
    # Convert all the images into standard size, and then devide 255 to normalize
    # The values of RGB, maximum is 255, minimum is 0
    
    # Process the train data
    trainTensors = Paths2Tensor(X_train).astype(np.float32) / 255
    
    # Process the validation data
    validationTensors = Paths2Tensor(X_validation).astype(np.float32) / 255
    
    # Process the test data
    testTensors = Paths2Tensor(X_test).astype(np.float32) / 255
    
    return trainTensors, validationTensors, testTensors
    
'''
files, originalTargets, dogTargets = Prepare("../../DogSpecies/Images/")

X_train, X_test, y_train, y_test, X_validation, y_validation = Split(files, dogTargets)

trainSensors, trainSensors, trainSensors = CreateTensors(X_train, X_validation, X_test)
'''