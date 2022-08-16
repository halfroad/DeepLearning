import random
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
import cv2
import os

def LoadFaces():
    
    # Load all the faces, return the path of each image, and form an array.
    facePaths = np.array(glob("../Inventory/lfw/*/*"))
    
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
    
    prefixLength = len("../Inventory/lfw/")
    
    # Load all folders under lfw, then remove the lfw/ prefix from foler name
    # Return an array via list inference to concat the names cut off
    faceNames = [item[prefixLength: ] for item in sorted(glob("../Inventory/lfw/*"))]
    
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
    
    prefixLength = len("../Inventory/lfw/")
    
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


def SwitchEnvironmentVariables():
    
    # Switch to directory FaceNet
    os.chdir("../FaceNet")
    # Output the current directory
    print(os.getcwd())

    # Set the environment variables, the subfolder src under FaceNet
    os.environ["PYTHONPATH"] = "../FaceNet/src"

    # Check the environment variables
    print(os.environ["PYTHONPATH"])
    
    '''
    
     python3 FaceNet/src/validate_on_lfw.py ../Inventory/Aligned  /Users/jinhui/Projects/DeepLearning/Chapters/11/Inventory/Models/20180402-114759.pb --distance_metric 1 --use_flipped_images --subtract_mean --use_fixed_image_standardization --lfw_pairs /Users/jinhui/Projects/DeepLearning/Chapters/11/11.1/FaceNet/data/pairs.txt
     
     python3 FaceNet/src/classifier.py TRAIN ../Inventory/CustomizedDatasets /Users/jinhui/Projects/DeepLearning/Chapters/11/Inventory/Models/20180402-114759.pb ../Inventory/Models/OwnedClassifier.pkl --image_size 160
     
     GeneratePairs.py: https://github.com/VictorZhang2014/facenet/blob/master/mydata/generate_pairs.py
     
     python3 FaceNet/src/train_softmax.py --logs_base_dir ../Inventory/LabeledFaceWild/Train/Logs/FaceNet/ --models_base_dir ../Inventory/LabeledFaceWild/Train/Models/FaceNet/ --data_dir ../Inventory/CustomizedDatasetsAligned/ --image_size 160 --model_def models.inception_resnet_v1 --lfw_dir ../Inventory/CustomizedDatasetsAligned/ --optimizer ADAM --learning_rate -1 --max_nrof_epochs 1 --keep_probability 0.8 --random_crop --random_flip --use_fixed_image_standardization --learning_rate_schedule_file FaceNet/data/learning_rate_schedule_classifier_casia.txt --weight_decay 5e-4 --embedding_size 512 --lfw_distance_metric 1 --lfw_use_flipped_images --lfw_subtract_mean --validation_set_split_ratio 0.05 --validate_every_n_epochs 5 --prelogits_norm_loss_factor 5e-4 --epoch_size 8 --lfw_batch_size 14 --lfw_pairs ../Inventory/Pairs/pairs.txt
     
     python3 FaceNet/src/validate_on_lfw.py ../Inventory/CustomizedDatasetsAligned ../Inventory/LabeledFaceWild/Train/Models/FaceNet/20220816-115035 --lfw_pairs ../Inventory/Pairs/pairs.txt --lfw_batch_size 2 --image_size 160 --distance_metric 1 --use_flipped_images --subtract_mean --use_fixed_image_standardization
     
    '''


facePaths = LoadFaces()
PreviewFace("../Inventory/Verification/4fe610510ua2fff22b15c2818b921ade.JPG")
CountFacesNumber()
PreviewRandomImage(facePaths)
RefineImages(facePaths)
