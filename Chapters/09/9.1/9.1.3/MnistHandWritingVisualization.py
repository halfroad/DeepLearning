import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(1, "../9.1.2/")
from MnistHandWritingRecognization import Prepare, Preprocess

def ShowFigure(imagesTrain):
    
    # Figure means the size of image to be plotted
    figure = plt.figure(figsize = (10, 10))
    
    # Plot and show the top 5 images of train data
    for i in range(5):
        
        ax = figure.add_subplot(1, 5, i + 1, xticks = [], yticks = [])
        
        ax.imshow(np.reshape(imagesTrain[i: i + 1], (28, 28)), cmap = "gray")
    
    figure = plt.figure(figsize = (10, 10))
    
    # Plot and show the top 5 images of train data
    for i in range(5):
        
        ax = figure.add_subplot(1, 5, i + 1, xticks = [], yticks = [])
        
        ax.imshow(np.reshape(imagesTrain[i + 2 * 12: i + 1 + 2 * 12], (28, 28)), cmap = "gray")
        
        
    plt.show()
        
# A method to visualize the image, a vector of image and figure object
def VisualizeInput(image):
    
    figure = plt.figure(figsize = (10, 10))
    ax = figure.add_subplot(111)
    
    # Plot and output the image
    ax.imshow(image, cmap = "gray")
    
    # Output the values of width and height for the ourpose of understanding how the computer to display the image legible
    width, height = image.shape
    
    # Convert the values of image to 0 ~ 1
    thresh = image.max() / 2.5

    # Iterate the rows
    for x in range(width):
        
        # Iterate the column
        for y in range(height):
            
            # Display the values of image on the x-y axis, make it perpendicular and center aligned
            ax.annotate(str(round(image[x][y], 2)), xy = (y, x),
                            horizontalalignment = "center",
                            verticalalignment = "center",
                            color = "white" if image[x][y] < thresh else "black")
            
    plt.show()
    
imagesTrain, labelsTrain, imagesTest, labelsTest = Prepare()
imagesTrain, imagesTest = Preprocess(imagesTrain, imagesTest)

#ShowFigure(imagesTrain)
    
VisualizeInput(np.reshape(imagesTrain[5: 6], (28, 28)))
