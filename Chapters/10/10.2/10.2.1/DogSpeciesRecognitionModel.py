from keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

import sys
sys.path.insert(1, "../../10.1/10.1.2/")
sys.path.insert(1, "../../10.1/10.1.3/")

from DogSpeciesRecognitionVisualization import Prepare, Split, classificationsNumber
from DogSpeciesRecognitionImagesNormalization import CreateTensors



def CreateModel(trainSensors, classificationsNumber):
    
    # Create the model of Sequnetial
    model = Sequential()
    
    # Add the input layer. The input layer should be the passed input_shape in order to represent the size of image. Depth is 16
    model.add(Conv2D(filters = 16, kernel_size = (2, 2), strides = (1, 1), padding = "same", activation = "relu", input_shape = trainSensors.shape[1: 1]))
    
    # Add the Maximum Pooling layer, size is 2 * 2, effetive scope is valid that means the data less than 2 * 2 will be dropped.
    model.add(MaxPool2D(pool_size = (2, 2)))
    
    # Add the Dropout layer, 20% network node will be dropped for the purpose of avoiding over fitting
    model.add(Dropout(0.2))
    
    # Add the Convolutional Layer, depth is 32, kernel size is 2 * 2, stride is 1 * 1. Effective scope same means the data less than the scope will be padding with 0
    model.add(Conv2D(filters = 32, kernel_size = (2, 2), strides = (1, 1), padding = "same", activation = "relu"))
    
    # Add tje Convolutional Layer, depth is 64
    model.add(Conv2D(filters = 64, kernel_size = (2, 2), strides = (1, 1), padding = "same", activation = "relu"))
    
    model.add(MaxPool2D(pool_size = (2, 2)))
    model.add(Dropout(0.2))
    
    # Add Global Average Pooling layer
    model.add(GlobalAveragePooling2D())
    
    # Add the Dropoutl layer, 50% network node will be dropped out
    mode.add(Dropout(0.5))
    
    # Add Output Layer, 120 species
    model.add(Dense(classificationsNumber, activation = "softmax"))
    
    # Print the summary of model architecture
    print(model.summary())
    
    
files, originalTargets, dogTargets = Prepare("../../DogSpecies/Images/")

X_train, X_test, y_train, y_test, X_validation, y_validation = Split(files, dogTargets)

trainSensors, trainSensors, trainSensors = CreateTensors(X_train, X_validation, X_test)
    
CreateModel(trainSensors, classificationsNumber)