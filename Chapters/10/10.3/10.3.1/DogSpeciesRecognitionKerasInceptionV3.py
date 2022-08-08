from keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions

def PlotBar(predictions):
    
    types = [prediction[1] for prediction in predictions]
    probs = [prediction[2] for prediction in predictions]
    
    plt.barh(np.arange(len(probs)), probs)
    
    _ = plt.yticks(np.arange(len(predictions)), types)
    
    plt.show()
    
def loadImaqge(imagePath):
    
    image = load_img(imagePath, target_size = (299, 299))
    
    array = img_to_array(image)
    array = np.expand_dims(array, axis = 0)
    
    return array

def PredictByInceptionV3(imagePath):
    
    # Predict directly by train model InceptionV3
    # Load the image
    
    array = loadImaqge(imagePath)
    # Preprocess the image
    array = preprocess_input(array)
    
    # Load the InceptionV3 train model
    model = InceptionV3(weights = "imagenet")
    # PRedict the dig species
    predictions = model.predict(array)
    # Decode the values of prediction, only top 5 will be displayed
    predictions = decode_predictions(predictions, top = 5)[0]
    
    # Draw the bar
    PlotBar(predictions)
    
    # Draw the original image and predicted possibility figure
    # Create a figure object
    figure, ax = plt.subplots()
    # Set the figure size
    figure.set_size_inches(5 ,5)
    
    # Grab the names of predicted species and its corresponding possibility
    figureTitle = "".join(["{}: {:.5f}%\n".format(prediction[1], prediction[2] * 100) for prediction in predictions])
    
    # Set the annotation text alongside the figure
    ax.text(1.01, 0.7, figureTitle, horizontalalignment = "left", verticalalignment = "bottom", transform = ax.transAxes)
    
    # Read the values from image
    image = plt.imread(imagePath)
    
    # Show the image on Axes
    ax.imshow(image)
    
    plt.show()
    
# Grab a image from train set
imagePath = "../../DogSpecies/Images/n02108000-EntleBucher/n02108000_93.jpg"
PredictByInceptionV3(imagePath)