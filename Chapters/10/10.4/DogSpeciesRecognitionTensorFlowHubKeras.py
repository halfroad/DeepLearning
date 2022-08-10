import tensorflow as tf
import keras
import tensorflow_hub as hub
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
import matplotlib.pyplot as plt
import numpy as np
import os

def DownloadImageNetDogs():
    
    path = keras.utils.get_file("Images", "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar", untar = True)
    
    #os.rmdir(path)
    
    print(path)

    
def Prepare():
    
    inputImageSize = (224, 224)
    
    imageDataGenerator = ImageDataGenerator(rescale = 1 / 255)
    imageFlow = imageDataGenerator.flow_from_directory("Images", target_size = inputImageSize, batch_size = 32, color_mode = "rgb", class_mode = "categorical")
    
    for imageBatch, labelBatch in imageFlow:
        
        print("Image batch shape: ", imageBatch.shape)
        print("Label batch shape: ", labelBatch.shape)
        
        break
    
    stepsPerEpoch = imageFlow.samples // imageFlow.batch_size
    
    print("There are {} image samples, Batch Size is {}".format(imageFlow.samples, imageFlow.batch_size))
    print("A comprehensive train requires {} times.".format(stepsPerEpoch))
        
    return imageFlow, inputImageSize, imageBatch, stepsPerEpoch

def CreateModel(imageFlow, inputImageSize):
    
    featureExtractorURL = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5"

    layers = [
        hub.KerasLayer(handle = featureExtractorURL, input_shape = inputImageSize + (3, ), trainable = False, name = "mobilenet_embedding"),
        tf.keras.layers.Dense(imageFlow.num_classes, activation="softmax")]
    model = tf.keras.Sequential(layers, name = "DogSpeciesFeatures")
        
    print(model.summary())
    
    return model


class CollectionBatchStats(tf.keras.callbacks.Callback):
        
        def __init__(self):
            
            self.batchLosses = []
            self.batchAccuracies = []
            
        def on_train_batch_end(self, batch, logs = None):
                
            self.batchLosses.append(logs["loss"])
            self.batchAccuracies.append(logs["accuracy"])

def TrainModel(model, imageFlow, stepsPerEpoch):
    
    model.compile(optimizer = keras.optimizers.Adam(), loss = "categorical_crossentropy", metrics = ["accuracy"])
    
    batchStats = CollectionBatchStats()
    
    model.fit((item for item in imageFlow), epochs = 5, steps_per_epoch = stepsPerEpoch, callbacks = [batchStats])
    
    PlotBatchStatistics(batchStats)
    
def PlotBatchStatistics(batchStats):
    
    plt.figure()
    
    plt.ylabel("Loss")
    plt.xlabel("Training Steps")
    
    # Show the pitch and lowest tide
    plt.ylim([0, 6])
    plt.plot(batchStats.batchLosses)
    
    plt.figure()
    
    plt.ylabel("Accuracy")
    plt.xlabel("Training Steps")
    
    plt.ylim([0, 1])
    plt.plot(batchStats.batchAccuracies)
    
    plt.show()
    
def Predict(model, imageFlow, imageBatch):
    
    # Acquire the sorted names of the dog species
    labelNames = sorted(imageFlow.class_indices.items(), key = lambda pair: pair[1])
    # Only return the recognized dog species
    labelNames = np.array([key.title() for key, value in labelNames])
    
    # Show the top 15
    print(labelNames[: 15])
    
    # Show the possibility of predicted dog species
    resultBatch = model.predict(imageBatch)
    
    # Associate the predicted with the maximum Probability with the dog species name
    maximumProbability = np.argmax(resultBatch, axis = -1)
    labelsBatch = labelNames[maximumProbability]
    
    # Find the array of the maximum Probability and the corresponding Probability of dog species name
    resultBatchProbabilities = [j[maximumProbability[i]] for i, j in enumerate(resultBatch)]
    
    print(labelsBatch[: 10])
    print(resultBatchProbabilities[: 10])
    
    return resultBatchProbabilities, labelsBatch, labelNames
    
def PlotDogs(imageBatch, resultBatchProbabilities, labelsBatch):
    
    # Size of the figure
    plt.figure(figsize = (10, 10))
    # The title of figure
    plt.suptitle("Model Predictions")
    
    # Iterate 9 times to show pictures
    for n in range(9):
        
        # Find the number n picture
        plt.subplot(3, 3, n + 1)
        # Show thre picture
        plt.imshow(imageBatch[n])
        # Set the title for the image, the title is Species Name: Probability
        prefix = len("N02085620-")
        # Convert the probability to percentage, and reserve the 2 digits after the decimal point
        probability = round(resultBatchProbabilities[n] * 100, 2)
        plt.title(labelsBatch[n][prefix: ] + ": " + str(probability) + "%")
        
        # Hide the axis
        plt.axis("off")
        
        plt.show()

# Load the image
def LoadImage(imagePath):
    
    # Load the image via path
    _image = image.load_img(imagePath, target_size = (224, 224))
    
    arrray = image.img_to_array(_image)
    arrary = np.expand_dims(arrary, axis = 0)
    
    return arrary

# Predict the dog species
def PredictDogSpecies(model, dogSpeciesNames, imagePath):
    
    # Load the image
    arrary = LoadImage(imagePath)
    # Image preprocess
    arrary = preprocess_input(array)
    # Predict
    predictions = model.predict(arrary)
    prediction = predictions[0]
    
    # Find the maximum index and maximum value
    def GetMaximum(prediction):
        
        maximumArgument = np.argmax(prediction)
        maximumValue = prediction[argumentMaximum]
        prediction = np.delete(prediction, argumentMaximum)
        
        return prediction, maximumArgument, maximumValue
    
    def GetTopThreePredictedMaximumIndexValues(prediction):
        
        preds, maximumArgument1, maximumValue1 = GetMaximum(prediction)
        preds, maximumArgument2, maximumValue2 = GetMaximum(preds)
        preds, maximumArgument3, maximumValue3 = GetMaximum(preds)
        
        top3MaximumArguments = np.array([maximumArgument1, maximumArgument2, maximumArgument3])
        top3MaximumValues = np.array([maximumValue1, maximumValue2, maximumValue3])
        
        return top3MaximumArguments, top3MaximumValues
    
    top3MaximumArguments, top3MaximumValues = GetTopThreePredictedMaximumIndexValues(prediction)
    titlePrefix = len("N02098413-")
    dogTitles = [dogSpeciesNames[index][titlePrefix: ] for index in top3MaximumArguments]
    
    print("Top 3 maximum values are: {}".format(top3MaximumValues))
    
    plt.barh(np.arange(3), top3MaximumValues)
    plt.yticks(np.arange(3), dogTitles)
    
    plt.show()
    
    # Create the object of figure
    figure, ax = plt.subplots()
    # Set the size of container
    figure.set_size_inches(5, 5)
    # Multiplify the maximum value 100 is the percetage
    top3MaximumValues *= 100
    
    # Concatenate the 3 maximum strings
    dogTitle = "{}: {:.2f}%\n".format(dogTitles[0], top3MaximumValues[0]) + \
            "{}: {:.2f}%\n".format(dogTitles[1], top3MaximumValues[1]) + \
            "{}: {:.2f}%\n".format(dogTitles[2], top3MaximumValues[2])
    # Add the strings of recognized value on the top right corner
    ax.text(1.01, 0.8, dogTitle, horizontalalignment = "left", verticalalignment = "bottom", transform = ax.transAxes)
    
    # Read the value content of image
    _image = plt.image.imread(imagePath)
    # Show the image on Axes
    ax.imshow(_image)
    
    plt.grid(False)
    
    plt.show()

# DownloadImageNetDogs()
imageFlow, inputImageSize, imageBatch, stepsPerEpoch = Prepare()
model = CreateModel(imageFlow, inputImageSize)

TrainModel(model, imageFlow, stepsPerEpoch)
# LoadPretrainModel(1)
resultBatchProbabilities, labelsBatch, labelNames = Predict(model, imageFlow, imageBatch)
PlotDogs(imageBatch, resultBatchProbabilities, labelsBatch)

imagePath = "ImageToBePredicted.jpg"
PredictDogSpecies(model, labelNames, imagePath)
