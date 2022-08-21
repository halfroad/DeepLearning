# https://github.com/21-projects-for-deep-learning/tf-pose-estimation.git
# https://github.com/gsethi2409/tf-pose-estimation
# https://medium.com/@gsethi2409/pose-estimation-with-tensorflow-2-0-a51162c095ba

'''

tf-pose-estimation/tf_pose/estimator.py

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
'''

import time
import numpy as np
import matplotlib.pyplot as plt
import glob
import sys

sys.path.append("../tf-pose-estimation/")

from PIL import Image


# Import tf_pose module
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path

def InferImages(paths):
    
    # Specify the name of the model, can be cmu or mobilenet_thin
    name = "cmu"    # "mobilenet_thin
    
    # Set the size ratio for output image
    resizeOutputRatio = 4.0
    # Set the width and height for the target window
    width = 375
    height = 667
    
    # Get the path of model via get_graph_path() method
    # Initialize the train model and target window size using class TfPoseEstimator
    model = get_graph_path(name)
    estimator = TfPoseEstimator(model, target_size = (width, height))
    
    for path in paths:
        
        # Load the image via PIL
        image = Image.open(path)
        # Convert the PIL image into NumPy array
        image = np.asarray(image)
        
        # Infer the image, return the key parts of body
        keyParts = estimator.inference(image, resize_to_default = (width > 0 and height > 0), upsample_size = resizeOutputRatio)
        image = TfPoseEstimator.draw_humans(image, keyParts, imgcopy = False)
        
        # Initialize a window of size 7 * 12
        figure, ax = plt.subplots(figsize = (7, 12))
        
        # Show the image
        ax.imshow(image)
    
    # Disable the grid
    plt.grid(False)
    
    plt.show()
    

files = glob.glob("../Inventory/Images/*.jpg")
InferImages(files)
