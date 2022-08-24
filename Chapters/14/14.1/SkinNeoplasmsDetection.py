from sklearn import datasets
from tqdm import tqdm
from matplotlib import image
from sklearn import model_selection

import numpy as np
import matplotlib.pyplot as plt

def AcquireImages():
    
    path = "../Inventory/DataSets"
    
    files = datasets.load_files(path)
    
    images = files["filenames"]
    targets = files["target"]
    targetNames = files["target_names"]
    
    print("images.shape = {}, images[: 20] = {}, targets.shape = {}, targets = {}, targetNames = {}.".format(images.shape, images[: 20], targets.shape, targets, targetNames))
    
    return images, targets, targetNames
    
def Visualize(images, targets, targetNames):
    
    # Use the default style for matplotlib when drawing
    plt.style.use("default")
    
    # Create 9 drawing objects, 3 rows and 3 columns
    figure, axes = plt.subplots(nrows = 3, ncols = 3)
    
    # Set the size of figure
    figure.set_size_inches(8, 7)
    
    # Select 9 numbers randomly, means 9 disease images
    randomNumbers = np.random.choice(len(images), 9)
    
    # Select 9 images from dataset, and its pathes
    randomimages = images[randomNumbers]
    
    index = 0
    
    # Row
    for row in range(3):
        
        for column in range(3):
            
            # Read the value from image
            _image = image.imread(randomimages[index])
            
            shape = _image.shape
            
            print("The shape of image number #{} is {}.".format(index, shape))
            
            # Grab the Axes object according to [row, column]
            ax = axes[row, column]
            
            ax.imshow(_image)
            
            target = targets[index]
            # Set the title with the name of skin cancer
            ax.set_xlabel(targetNames[target])
            
            index += 1
            
    plt.show()

def Split(images, targets):

    imagesTrain, imagesTest, cancersTrain, cancersTest = model_selection.train_test_split(images, targets, test_size = 0.2)
    
    # Bisect the test set into test set and validation set
    half = int(len(imagesTest) / 2)
    
    imagesValidation = imagesTest[: half]
    cancersValidation = cancersTest[: half]
    
    imagesTest = imagesTest[half:]
    cancersTest = cancersTest[half:]
    
    print("imagesTrain.shape = {}, cancersTrain.shape = {},".format(imagesTrain.shape, cancersTrain.shape))
    print("imagesValidation.shape = {}, cancersValidation.shape = {},".format(imagesValidation.shape, cancersValidation.shape))
    print("imagesTest.shape = {}, cancersTest.shape = {},".format(imagesTest.shape, cancersTest.shape))
    
    return imagesTrain, imagesTest, imagesValidation, cancersValidation, cancersTrain, cancersTest
    
images, targets, targetNames = AcquireImages()
Visualize(images, targets, targetNames)

imagesTrain, imagesTest, imagesValidation, cancersValidation, cancersTrain, cancersTest = Split(images, targets)
