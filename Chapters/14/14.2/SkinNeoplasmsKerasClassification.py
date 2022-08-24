from keras.utils import load_img, img_to_array
from PIL import ImageFile
from tqdm import tqdm

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

import numpy as np
import sys

sys.path.append("../14.1")
from SkinNeoplasmsDetection import AcquireImages, Split

def Tensorize(imagePath):
    
    '''
    Convert the specific image into the Convolutional Neural Network friendly shape (1, 224, 224, 3)
    '''
    
    # Load the image with load_img() in PIL module, a PIL object will be returned
    _image = load_img(imagePath, target_size = (224, 224, 3))
    # Convert the PIL image into array
    array = img_to_array(_image)
    
    # Reshape the array to (1, 224, 224, 3), a 4 dimensions tensor
    
    return np.expand_dims(array, axis = 0)

def ImagesTensorize(imagePaths):
    
    '''
    Convert all the images within the array into a value type
    '''
    
    # Use the tqdm to indicate the progress by passing an array of image paths
    tensors = [Tensorize(path) for path in tqdm(imagePaths)]
    
    # Stack the objects with perpendicular direction
    return np.vstack(tensors)

def CreateModel(targetNames, trainTensors):
    
    # Shape of image
    inputShape = trainTensors[0].shape
    # Number of classifications
    classificationsNumber = len(targetNames)
    
    # Create model of Sequential
    model = Sequential()
    
    # Add Input Layer, the input_shape (indicates the size of image) shall be passed to Input Layer
    # Depth of Input Layer is 16
    model.add(Conv2D(filters = 16, kernel_size = (1, 1), strides = (1, 1), padding = "same", activation = "relu", input_shape = inputShape))
    # Add MaxPooling Layer, the size of Convolutional layer is 1 * 1, effective padding is valid default
    model.add(MaxPooling2D(pool_size = (1, 1)))
    # Add the Dropout layer, drop 50% in each network node for avoiding Overfitting
    model.add(Dropout(0.5))
    
    # Add Convolutional Layer, depth is 32, kernel siz eis 1 * 1, strides is 1 * 1, use ReLU to activate the neural network
    model.add(Conv2D(filters = 32, kernel_size = (1, 1), strides = (1, 1), padding = "same", activation = "relu"))
    model.add(MaxPooling2D(pool_size = (1, 1)))
    model.add(Dropout(0.3))
    
    # Add Convolutional Layer, depth is 64
    model.add(Conv2D(filters = 64, kernel_size = (1, 1), strides = (1, 1), padding = "same", activation = "relu"))
    model.add(MaxPooling2D(pool_size = (1, 1)))
    model.add(Dropout(0.2))
    
    # Add Convolutional Layer, depth is 128
    model.add(Conv2D(filters = 128, kernel_size = (1, 1), strides = (1, 1), padding = "same", activation = "relu"))
    model.add(MaxPooling2D(pool_size = (1, 1)))
    model.add(Dropout(0.2))
    
    # Add Average Pooling Layer, process the spatial data
    model.add(GlobalAveragePooling2D())
    # Add the Dropout Layer, drop out 50%
    model.add(Dropout(0.5))
    
    # Add Output Layer, output 3 classifications
    model.add(Dense(classificationsNumber, activation = "softmax"))
    
    # Print the architecture of Network Neural
    print(model.summary())
    
    return model
    
def TrainModel(model, cancersTrain, cancersValidation, trainTensors, validationTensors):
    
    model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
    
    # If the sparse_categorical_crossentropy is used for Loss Function here, following one-hot encoding is needed anymore.
    # If categorical_crossentropy is used, following statements shall be commented out
    # from keras import utils
    # cancersTrain = utils.to_categorical(cancersTrain)
    # cancersValidation = utils.to_categorical(cancersValidation)
    # cancersTest = utils.to_categorical(cancersTest)
    
    # Create the object of CheckPoint
    checkPoint = ModelCheckpoint(filepath = "../Inventory/Models/Saved/SkinCancer.BestWeights.hdf5", verbose = 1, save_best_only = True)
    
    epochs = 10
    
    # Train the model
    model.fit(trainTensors, cancersTrain,
              validation_data = (validationTensors, cancersValidation),
              epochs = epochs,
              batch_size = 20,
              callbacks = [checkPoint], verbose = 1)
    
# Avoid the I/O error when reading image via PIL, set the Image Truncation
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Convert all the images into value type based ones,
# then devide 255 to normalize (Convert the RGB (0 ~ 255) to 0 ~ 1 for easing the handling of CNN)
# RGB is ranged in 0 ~ 255

imagePaths, targets, targetNames = AcquireImages()

imagesTrain, imagesTest, imagesValidation, cancersValidation, cancersTrain, cancersTest = Split(imagePaths, targets)

# Handle the Train Set
trainTensors = ImagesTensorize(imagesTrain).astype(np.float32) / 255

# Handle the Validation Set
validationTensors = ImagesTensorize(imagesValidation).astype(np.float32) / 255

# Handle the Test Set
testTensors = ImagesTensorize(imagesTest).astype(np.float32) / 255

model = CreateModel(targetNames, trainTensors)

TrainModel(model, cancersTrain, cancersValidation, trainTensors, validationTensors)
