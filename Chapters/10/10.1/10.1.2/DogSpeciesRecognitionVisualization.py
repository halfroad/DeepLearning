from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import image

# 120 species in all
classificationsNumber = 120

# Load the dataset

def Prepare(path):
    
    # Load the files via the method load_files() on sklearn
    # This method returns a object of dictionary comprised with releative path and its serial number
    data = load_files(path)
    
    # Convert the paths into Numpy object
    files = np.array(data["filenames"])
    # Sort the images by serial numbers
    originalTargets = np.array(data["target"])
    
    # Convert the file serial numbers into binary classified matrix (AKA one-hot encoding) via to_categorical()
    dogTargets = np_utils.to_categorical(originalTargets, classificationsNumber)
    
    # Return the paths of all the images, serial numbers and binary classfied matrix
    return files, originalTargets, dogTargets

def Split(files, dogTargets):
    
    # Select top 9000 for potnetial performance issue. Comment out this statement if there is no such issue
    files = files[: 9000]
    dogTargets = dogTargets[: 9000]
    
    # Split the train data and test data
    X_train, X_test, y_train, y_test = train_test_split(files, dogTargets, test_size = 0.2)
    
    # Split the test data into test data and validation data
    halfTestCount = int(len(X_test) / 2)
    
    X_validation = X_test[: halfTestCount]
    y_validation = y_test[: halfTestCount]
    
    X_test = X_test[halfTestCount:]
    y_test = y_test[halfTestCount:]
    
    print("X_train.shape = {}, X_test.shape = {}".format(X_train.shape, X_test.shape))
    print("y_train.shape = {}, y_test.shape = {}".format(y_train.shape, y_test.shape))
    print("X_validation.shape = {}, y_validation.shape = {}".format(X_validation.shape, y_validation.shape))
    
    return X_train, X_test, y_train, y_test, X_validation, y_validation

# Randomize 9 images to be displayed
def Visualize(X_train, prefixLength):
    
    # Set the default style
    plt.style.use("default")
    
    # Create 9 objects for plot, 3 rows * 3 columns
    figure, axes = plt.subplots(nrows = 3, ncols = 3)
    
    # Set the canvas size
    figure.set_size_inches(10, 9)
    
    # Randomly select 9 images, that is 9 dog species (May be duplicate, it is not the same each time)
    random9Numbers = np.random.choice(len(X_train), 9)
    
    # Select 9 images from train data
    random9Images = X_train[random9Numbers]
    
    print(random9Images)
    
    imageNames = []
    
    for path in random9Images:
        
        imageName = path[prefixLength:]
        imageName = imageName[: imageName.find("/")]
        
        imageNames.append(imageName)
        
    index = 0
    
    # Row
    for row in range(3):
        
        for column in range(3):
            
            # Read the content from image
            _image = image.imread(random9Images[index])
            # Create the Axes object by [row, column]
            ax = axes[row, column]
            
            # Show the image on Axes
            ax.imshow(_image)
            
            # Set the dog species on Axes
            ax.set_xlabel(imageNames[index])
            
            # Auto increase index
            index += 1
    
    plt.show()
    

def AnalyzeImages(paths):
    
    # Iterate the paths, read each image and get the dimension of the image
    # The last returned shape of image will be stored in shapes list
    
    _shapes = []
    
    for path in paths:
        
        shape = image.imread(path).shape
        
        if len(shape) == 3:
            _shapes.append(shape)
            
    shapes = np.asarray(_shapes)
    
    print("There are {} images".format(len(shapes)))
    print("The dimensions of the 3 images randomly selected: {}".format(shapes[np.random.choice(len(shapes), 3)]))
    
    dogsMeanWidth = np.mean(shapes[:, 0])
    dogsMeanHeight = np.mean(shapes[:, 1])
    
    print("Mean width of dog image is {:.1f}, mean height of dog images is {:.1f}.".format(dogsMeanWidth, dogsMeanHeight))
    
    # show the intent of the width and height for the images
    # Parameter 1: width of the image
    # Parameter 2: height of the image
    # Parameter 3: Notate the data with "o"
    plt.plot(shapes[:, 0], shapes[:, 1], "o")
    
    # Set the title
    plt.title("The Image Sizes of All Categories of Dog")
    
    # Set the label for X axis
    plt.xlabel("Image Width")
    
    # Set the label for Y axis
    plt.ylabel("Image Height")
    
    plt.show()
        
    # Distribution for Width
    plt.hist(shapes[:, 0], bins = 100)
    plt.title("Dog Images With Width Disbribution")
    
    plt.show()
    
    plt.hist(shapes[:, 1], bins = 100)
    plt.title("Dog Images With Height Disbribution")
    
    plt.show()

'''
# Load the dataset
files, originalTargets, dogTargets = Prepare("../../DogSpecies/Images/")

# Load the list of dog species
# Glob is file operation related module. Return the path of file or directory via specific match pattern
# Here the operation is to return all the directories under Images
# Iterate the paths via list deduction, and truncate the string of species name of the dog
prefixLength = len("Images/n02085620-")
dogNames = [item[prefixLength: ] for item in sorted(glob("../../DogSpecies/Images/*"))]

print("There are {} dog species in total.".format(len(dogNames)))
print("There are {} dog images in total.\n".format(len(files)))

print(dogNames[: 5])
print(files[: 5])
print(originalTargets[: 10])
print(dogTargets[: 3])

X_train, X_test, y_train, y_test, X_validation, y_validation = Split(files, dogTargets)

Visualize(X_train, prefixLength)
AnalyzeImages(files)
'''