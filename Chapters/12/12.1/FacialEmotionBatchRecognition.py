from glob import glob
from PIL import ImageFont, ImageDraw, Image

import matplotlib
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random

from FacialEmotionRecognition import LoadImage, LoadModel

# 7 classifications of facial emotions
classifications = ["愤怒", "厌恶", "害怕", "高兴", "悲伤", "惊喜", "平淡"]

def PlotFacialEmotions(model, paths):
    
    # Draw the facial matrix
    # Randomize the paths
    random.shuffle(paths)
    
    # Load the font Songti
    fontPath = "../../../Universal/Fonts/Songti.ttc"
    # fontPath = "Songti.ttc"
    font = ImageFont.truetype(fontPath, 120)
    
    rows = 2
    columns = 4
    
    # Create a figure with 2 rows, 4 columns
    figure, axes = plt.subplots(nrows = rows, ncols = columns)
    
    # Set the width and height for whole figure
    figure.set_size_inches(12, 6)
    
    index = 0
    
    # Iterate 2 rows
    for row in range(rows):
        
        # Iterate 4 columns
        for column in range(columns):
            
            # Recognize the image and return the probabilities via load_img
            image = LoadImage(paths[index])
            probabilites = model.predict(image)
            
            # Grab the categorical probabilities
            probabilites = probabilites[0]
            
            # Grab the index of maximum probability
            maximum = np.argmax(probabilites)
            # Grab the value of maximum probability
            probability = probabilites[maximum]
            
            # Grab the emotion entry via maximum index
            emotion = classifications[maximum]
            # Concatenate the emtion and probability
            emotion = emotion + ": " + str(round(probability * 100, 2)) + "%"
            
            # Display the emotion and probability on top left corner of the image
            # Read the image from given path
            _image = matplotlib.image.imread(paths[index])
            # Convert the image into RGB mode
            pilImage = Image.fromarray(cv2.cvtColor(_image, cv2.COLOR_BGR2RGB))
            
            # Create the object of plot
            drawable = ImageDraw.Draw(pilImage)
            # Draw the emtion and probability on the top left corner with the x = 30 and y = 5
            drawable.text((30, 0), emotion, font = font, fill = (0, 0, 255))
            # Then convert back the image from RGB to BGR mode
            finalImage = cv2.cvtColor(np.asarray(pilImage), cv2.COLOR_RGB2BGR)
            
            # Grab the Axes object of matplotlib
            ax = axes[row, column]
            
            # Show the image
            ax.imshow(finalImage)
            
            index += 1
            
    plt.show()
    

    
paths = glob("../Inventory/Emotions/*.jpg")

model = LoadModel()
PlotFacialEmotions(model, paths)
