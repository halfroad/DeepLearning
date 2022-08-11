import random
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
import cv2

def LoadFaces():
    
    # Load all the faces, return the path of each image, and form an array.
    facePaths = np.array(glob("../Data/lfw/*/*"))
    
    # Disorder the array of facePaths via shuffle()
    random.shuffle(facePaths)
    
    print(facePaths[: 10])
    print("facePaths.shape = {}.".format(facePaths.shape))
    
    return facePaths

def PreviewFace(facePath):
    
    # Set the default style
    plt.style.use("default")
    
    # Initialize the model object of Face Recognition from OpenCV
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
    
    # Load the color channels (the order is BGR) from image
    # Select an image randomly to testify
    _image = cv2.imread(facePath)
    # Make the image to grey out
    gray = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
    # Find the face image
    faces = faceCascade.detectMultiScale(gray)
    
    # Number of faces detected
    print("Number of faces detected: ", len(faces))
    
    # Grab the detected rectangles of the faces
    for (x, y, w, h) in faces:
        
        # Draw the rectangle for the detected faces
        # Parameter 0: Image object
        # Parameter 1: (x, y) Start coordinate
        # Parameter 2: (x, y) Detected maximum coordinate o the face
        # Parameter 3: Color of border. Since the order is BGR, so the 3rd 255 means red
        # Parameter 4: Width of border
        cv2.rectangle(_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    # Convert the BGR image to RGB image so that the image can be printed
    rgbImage = cv2.cvtColor(_image, cv2.COLOR_BGR2RGB)
    
    # Display the rectangle of the recognited face
    plt.imshow(rgbImage)
    
    plt.show()
    
def CountFacesNumber():
    
    prefixLength = len("lfw/")
    
    # Load all folders under lfw, then remove the lfw/ prefix from foler name
    # Return an array via list inference to concat the names cut off
    faceNames = [item[prefixLength: ] for item in sorted(glob("lfw/*"))]
    
    print("Number of total faces is {}.".format(len(faceNames)))
    print(faceNames[: 10])
    
def PreviewRandomImage(facePaths):
    
    # Create 9 figure objects, 3 rows and 3 columns
    figure, axes = plt.subplots(nrows = 3, ncols = 3)
    
    # Set the size of figure
    figure.set_size_inches(10, 10)
    
    # Select 9 images randomly that means 9 face images (Maybe duplicate and not the same each time)
    random9Numbers = np.random.choice(len(facePaths), 9)
    
    # Select 9 images from dataset
    random9Images = facePaths[random9Numbers]
    
    print(random9Images)
    
    # Grab the names according to face which is 9 selected images randomly
    
    names = []
    
    prefixLength = len("lfw/")
    
    for path in random9Images:
        
        name = path[prefixLength: ]
        name = name[: name.find("/")]
        
        names.append(name)
        
    index = 0
    
    # Row
    for row in range(3):
        # Column
        for column in range(3):
            
            # Read the values from image
            _image = image.imread(random9Images[index])
            # Grab the Axes object according to [row, column]
            ax = axes[row, column]
            
            # Show the image on Axes
            ax.imshow(_image)
            
            # Set the name on the figure
            ax.set_xlabel(names[index])
            
            # Increase the index
            index += 1
            
    plt.show()
    
def RefineImages(facePaths):
    
    faceShapes = []
    
    for path in facePaths:
        
        shape = image.imread(path).shape
        
        if len(shape) == 3 and shape[0] == 250 and shape[1] == 250 and shape[2] == 3:
            faceShapes.append(shape)
        else:
            print("Observed an abnormal face image, path is: {}.".format(path))
    
    faceShapes = np.asarray(faceShapes)
    
    print("Total number is {}.".format(len(faceShapes)))
    print("The shapes of randomy selected 3 images are {}.".format(faceShapes[np.random.choice(len(faceShapes), 3)]))
    
facePaths = LoadFaces()
# PreviewFace("../Images/4fe610510ua2fff22b15c2818b921ade.JPG")
# CountFacesNumber()
PreviewRandomImage(facePaths)
CompareImages(facePaths)