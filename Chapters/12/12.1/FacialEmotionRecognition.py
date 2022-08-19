import numpy as np

import keras
from keras import utils
from keras.models import Sequential
from keras.models import model_from_json
from keras.models import load_model
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, activation, Dropout, Flatten
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

import glob


# Define the width and height for image
width = 48
height = 48

# 7 classifications of facial emotions
classifications = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    
    
def LoadFromDisk():
    
    with open("../Inventory/fer2013/fer2013.csv") as file:
        
        # Read all the lines
        content = file.readlines()
        
    # Convet the content int NumPy matrix
    lines = np.array(content)
    instancesNumber = lines.size
    
    print("Number of instances: {}".format(instancesNumber))
    print("Length of instances: {}".format(len(lines[1].split(",")[1].split(" "))))
    
    return instancesNumber, lines
    

def Split(instancesNumber, lines):
    
    classificationsNumber = len(classifications)
    
    # Train Set, Test Set, Validation Set
    imagesTrain, emotionsTrain, imagesValidation, emotionsValidation, imagesTest, emotionsTest = [], [], [], [], [], []
    
    # Split the set
    for i in range(1, instancesNumber):
        
        try:
            
            # Read a line
            emotion, image, usage = lines[i].split(",")
                        
            # Seperate the image with SPACE into array
            array = image.split(" ")
            # Convert the array into NumPy array
            pixels = np.array(array, np.float32)
            
            # One-Hot Encodes the facial emotion
            emotion = utils.to_categorical(emotion, classificationsNumber)
            
            # Add the entry into Train Set if it is Train Data
            if "Training" in usage:
                
               imagesTrain.append(pixels)
               emotionsTrain.append(emotion)
               
            elif "PublicTest" in usage:
                
                imagesTest.append(pixels)
                emotionsTest.append(emotion)
            
        except:
            print("", end = "")

    # Bisect the Test Set for Validation Set
    half = int(len(imagesTest) / 2)
    
    imagesValidation = imagesTest[: half]
    emotionsValidation = emotionsTest[: half]
    
    imagesTest = imagesTest[half: ]
    emotionsTest = emotionsTest[half: ]
    
    return imagesTrain, emotionsTrain, imagesValidation, emotionsValidation, imagesTest, emotionsTest, classificationsNumber

def Preprocess(imagesTrain, emotionsTrain, imagesValidation, emotionsValidation, imagesTest, emotionsTest):
    
    # Convert the values on Train set into float32, and create the NumPy array to hold the float32
    imagesTrain = np.array(imagesTrain, np.float32)
    emotionsTrain = np.array(emotionsTrain, np.float32)
    
    # Convert the values on Validation set into float32, and create the NumPy array to hold the float32
    imagesValidation = np.array(imagesValidation, np.float32)
    emotionsValidation = np.array(emotionsValidation, np.float32)
    
    # Convert the values on Test set into float32, and create the NumPy array to hold the float32
    imagesTest = np.array(imagesTest, np.float32)
    emotionsTest = np.array(emotionsTest, np.float32)
    
    # Normalize the input image values, and converted the values into 0 ~ 1 ones
    imagesTrain /= 255
    imagesValidation /= 255
    imagesTest /= 255
    
    # Reshape the Train set with the shape (batch_size, height, width, channels) 4-dimensions array
    imagesTrain = imagesTrain.reshape(imagesTrain.shape[0], height, width, 1)
    imagesTrain = imagesTrain.astype(np.float32)
    
    # Reshape the Validation set with the shape (batch_size, height, width, channels) 4-dimensions array
    imagesValidation = imagesValidation.reshape(imagesValidation.shape[0], height, width, 1)
    imagesValidation = imagesValidation.astype(np.float32)
    
    # Reshape the Validation set with the shape (batch_size, height, width, channels) 4-dimensions array
    imagesTest = imagesTest.reshape(imagesTest.shape[0], height, width, 1)
    imagesTest = imagesTest.astype(np.float32)
    
    # Print the arrays
    print("imagesTrain = {}, emotionsTrain = {}".format(imagesTrain.shape, emotionsTrain.shape))
    print("imagesValidation = {}, emotionsValidation = {}".format(imagesValidation.shape, emotionsValidation.shape))
    print("imagesTest = {}, emotionsTest = {}".format(imagesTest.shape, emotionsTest.shape))
    
    return imagesTrain, emotionsTrain, imagesValidation, emotionsValidation, imagesTest, emotionsTest

def CreateModel(classificationsNumber):
    
    # Create the model instance of Sequential
    model = Sequential()
    
    # Add the 1st Convolutional Layer, the input_shape is the shape of image
    model.add(Conv2D(64, (5, 5), activation = "relu", input_shape = (width, height, 1)))
    model.add(MaxPooling2D(pool_size = (5, 5), strides = (2, 2)))
    model.add(Dropout(0.5))
    
    # Add the 2nd Convolutional Layer
    model.add(Conv2D(64, (3, 3), activation = "relu"))
    model.add(Conv2D(64, (3, 3), activation = "relu"))
    model.add(AveragePooling2D(pool_size = (3, 3), strides = (2, 2)))
    model.add(Dropout(0.5))
    
    # Add the 3rd Convolutional Layer
    model.add(Conv2D(128, (3, 3), activation = "relu"))
    model.add(Conv2D(128, (3, 3), activation = "relu"))
    model.add(AveragePooling2D(pool_size = (3, 3), strides = (2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    
    # Add 1024 Fully-Connected Layer
    model.add(Dense(1024, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation = "relu"))
    model.add(Dropout(0.2))
    
    # Add the Output Layer
    model.add(Dense(classificationsNumber, activation = "softmax"))
    
    # Print the summary of model
    print(model.summary())
    
    return model

def TrainModel(model, imagesTrain, emotionsTrain, imagesValidation, emotionsValidation):
    
    # Batch size each time
    batch_size = 256
    
    # The number of iterations
    epochs = 20
    
    # Create the Image Data Generator (Strengthener)
    imageDataGenerator = ImageDataGenerator()
    
    # iterator created after the image enhanced
    trainGenerator = imageDataGenerator.flow(imagesTrain, emotionsTrain, batch_size = batch_size)
    
    # Compile the model, use the Categorical Cross Entrophy as loss function, Adam as optimizer, and use accuracy to measure the result
    model.compile(loss = "categorical_crossentropy", optimizer = keras.optimizers.Adam(), metrics = ["accuracy"])
    
    steps_per_epoch = int(len(imagesTrain) / batch_size)
    
    # Train the model
    history = model.fit(trainGenerator, steps_per_epoch = steps_per_epoch, epochs = epochs, validation_data = (imagesValidation, emotionsValidation), verbose = 1)
    
    return history

def PlotHistory(history):
    
    # Draw the trends of loss and accuracy when training and verifying
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    
    plt.title("Model Accuracy")
    
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    
    plt.legend(["Train", "Test"], loc = "upper left")
    
    plt.show()
    
    # Draw the trends of loss and accuracy when training and verifying
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    
    plt.title("Model Accuracy")
    
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    
    plt.legend(["Train", "Test"], loc = "upper left")
    
    plt.show()

def EvaluateModel(model, imagesTrain, emotionsTrain, imagesTest, emotionsTest):
    
    trainScore = model.evaluate(imagesTrain, emotionsTrain, verbose = 0)
    
    print("Train Loss = {}".format(trainScore[0]))
    print("Train Accuracy = {}".format(trainScore[1]))
    
    testScore = model.evaluate(imagesTest, emotionsTest, verbose = 0)
    
    print("Test Loss = {}".format(testScore[0]))
    print("Test Accuracy = {}".format(testScore[1]))
    
def StoreModel(model):
    
    # Serialize the architecture of model, and store the model into JSON object
    modelJson = model.to_json()
    
    # Store the model into local disk
    with open("../Inventory/Models/FacialExpressionRecognitionModelArchitecture.json", "w") as jsonFile:
        
        jsonFile.write(modelJson)
        
    # Serialize the weights into a HDF5 (Hierarchical Data  Format) file
    model.save_weights("../Inventory/Models/FacialExpressionRecognitionModelWeights.h5")
    
def LoadModel():
    
    # Load the architecture of model
    with open("../Inventory/Models/FacialExpressionRecognitionModelArchitecture.json", "r") as jsonFile:
        
        json = jsonFile.read()
        model = model_from_json(json)
        
        model.load_weights("../Inventory/Models/FacialExpressionRecognitionModelWeights.h5")
        
        # Compile the model, use the Categorical Cross Entrophy as loss function, Adam as optimizer, and use accuracy to measure the result
        model.compile(loss = "categorical_crossentropy", optimizer = keras.optimizers.Adam(), metrics = ["accuracy"])
        
        return model
    
def LoadImage(path):
    
    # Load the RGB image with gray scale, adn resize the image into 48 * 48
    _image = utils.load_img(path, color_mode = "grayscale", target_size = (width, height))
    
    # Convert the image into array
    array = utils.img_to_array(_image)
    
    # Reshape the array to 4 dimensions
    array = np.expand_dims(array, axis = 0)
    
    # Normalize the image
    array /= 255
    
    return array

def PlotAnalyzeEmotion(probabilities, classifications):
    
    # Draw the bar to show the probabilities for classifications
    vertical = np.arange(len(classifications))
    
    plt.bar(vertical, probabilities, align = "center", alpha = 0.5)
    plt.xticks(vertical, classifications)
    plt.ylabel("Percentage")
    plt.title("Emotion Recognized")
    
    plt.show()
    
def Predict(model, path):
    
    # Load the image
    _image = LoadImage(path)
    # Recognition propabilities
    probabilites = model.predict(_image)
    # Define the classifications
    PlotAnalyzeEmotion(probabilites[0], classifications)
    
    
def DisplayImage(path, grayScale = False, resize = False):
    
     # Read the image from local disk to memory
    if grayScale:
        
        if resize:
   
            _image = utils.load_img(path, color_mode = "grayscale", target_size = (width, height))

            # show the image
            plt.imshow(_image)
            
        else:
            
             _image = utils.load_img(path, color_mode = "grayscale")
             
             # show the image
             plt.imshow(_image)
            
    else:
        _image = utils.load_img(path)

        # show the image
        plt.imshow(_image)
        
    plt.show()

instancesNumber, lines = LoadFromDisk()
imagesTrain, emotionsTrain, imagesValidation, emotionsValidation, imagesTest, emotionsTest, classificationsNumber = Split(instancesNumber, lines)
imagesTrain, emotionsTrain, imagesValidation, emotionsValidation, imagesTest, emotionsTest = Preprocess(imagesTrain, emotionsTrain, imagesValidation, emotionsValidation, imagesTest, emotionsTest)

model = CreateModel(classificationsNumber)
history = TrainModel(model, imagesTrain, emotionsTrain, imagesValidation, emotionsValidation)
EvaluateModel(model, imagesTrain, emotionsTrain, imagesTest, emotionsTest)

StoreModel(model)

PlotHistory(history)


DisplayImage("../Inventory/Verification/victor_test.jpeg")
DisplayImage("../Inventory/Verification/victor_test.jpeg", grayScale = True)

model = LoadModel()

files = glob.glob("../Inventory/Emotions/*")

for file in files:
    
    DisplayImage(file)
    Predict(model, file)
