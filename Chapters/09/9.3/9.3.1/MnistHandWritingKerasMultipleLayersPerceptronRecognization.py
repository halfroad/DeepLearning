import keras

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import RMSprop

# Prepare the parameters
# Size of samples for each training
batchSize = 128
    
# MNIST only has 10 classes, 0 - 9
classesNumber = 10
    
# Train all the spicemens, iterate 20 trainings perpetually
epochs = 20
    
# Width and height of image
imageSize = 28 * 28

def Prepare():
    
    # Download and read the MNIST datasets
    (imagesTrain, labelsTrain), (imagesTest, labelsTest) = mnist.load_data()
    
    # Split 5000 for validation set
    validationLength = 5000
    imagesLength = imagesTrain.shape[0]
    
    trainLength = imagesLength - validationLength
    
    # Validation Set
    imagesValidation = imagesTrain[trainLength:]
    labelsValidation = labelsTrain[trainLength:]
    
    # Train Set
    imagesTrain = imagesTrain[: trainLength]
    labelsTrain = labelsTrain[: trainLength]
    
    # Convert the Train Set, Validation Set, Test Set to be vectorized image
    imagesTrain = imagesTrain.reshape(imagesTrain.shape[0], imageSize)
    imagesValidation = imagesValidation.reshape(imagesValidation.shape[0], imageSize)
    imagesTest = imagesTest.reshape(imagesTest.shape[0], imageSize)
    
    # Convert the Train Set, Validation Set and Test Set to float32
    imagesTrain = imagesTrain.astype("float32")
    imagesValidation = imagesValidation.astype("float32")
    imagesTest = imagesTest.astype("float32")
    
    # Convert the Train Set, Validation Set, Test Set to be the value between 0 - 1, normalization
    imagesTrain /= 255
    imagesValidation /= 255
    imagesTest /= 255
    
    # One Hoted the labels of Train Set, Validation Set and Test Set by to_categorical()
    labelsTrain = keras.utils.to_categorical(labelsTrain, classesNumber)
    labelsValidation = keras.utils.to_categorical(labelsValidation, classesNumber)
    labelsTest = keras.utils.to_categorical(labelsTest, classesNumber)
    
    # Show the shape of sets
    print("Shape of Train Images: {}".format(imagesTrain.shape))
    print("Shape of Train Labels: {}".format(labelsTrain.shape))
    print("Shape of Validation Images: {}".format(imagesValidation.shape))
    print("Shape of Validation Labels: {}".format(labelsValidation.shape))
    print("Shape of Test Images: {}".format(imagesTest.shape))
    print("Shape of Test Labels: {}".format(labelsTest.shape))
    
    return imagesTrain, labelsTrain, imagesValidation, labelsValidation, imagesTest, labelsTest

def CreateModel():
    
    # Create a model
    model = Sequential()
    
    # Add an input layer, the size of layer is the size of image, activation function is the ReLU (Rectified Linear Unit)
    # input_shape is mandatory, its value is the size image
    model.add(Dense(512, activation = "relu", input_shape = (imageSize, )))
    
    # Add the layer Dropout
    model.add(Dropout(0.2))
    
    # Add 512 full-connected layers, and use ReLU activation
    model.add(Dense(512, activation = "relu"))
    
    # Add the layer Dropout
    model.add(Dropout(0.2))
    
    # Add  the layer of output, the number of output classifications is 10, use the multiple classes and classifications activation fucntion - softmax
    model.add(Dense(classesNumber, activation = "softmax"))
    
    # Preview the summary of model
    print(model.summary())
    
    return model
    
def TrainModel():
    
    model = CreateModel()
    
    # Compile the model
    model.compile(loss = "categorical_crossentropy", optimizer = RMSprop(), metrics = ["accuracy"])
    
    imagesTrain, labelsTrain, imagesValidation, labelsValidation, imagesTest, labelsTest = Prepare()
    # Train the model
    model.fit(imagesTrain, labelsTrain, epochs = epochs, batch_size = batchSize, verbose = 1, validation_data = (imagesValidation, labelsValidation))
    
TrainModel()