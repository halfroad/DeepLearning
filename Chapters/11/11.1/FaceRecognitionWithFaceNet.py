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
    
    
facePaths = LoadFaces()
PreviewFace("../Images/4fe610510ua2fff22b15c2818b921ade.JPG")