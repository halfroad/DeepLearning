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
import cv2

sys.path.append("../Exclusion/tf-pose-estimation/")

from PIL import Image
from moviepy.editor import VideoFileClip

# Import tf_pose module
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path


def InferImages(imagePath):
    
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
    
    # Load the image via PIL
    image = Image.open(imagePath)
    # Convert the PIL image into NumPy array
    image = np.asarray(image)
        
    # Infer the image, return the key parts of body
    keyParts = estimator.inference(image, resize_to_default = (width > 0 and height > 0), upsample_size = resizeOutputRatio)
    image = TfPoseEstimator.draw_humans(image, keyParts, imgcopy = False)
        
    '''

    # Initialize a window of size 7 * 12
    figure, ax = plt.subplots(figsize = (7, 12))
        
    # Show the image
    ax.imshow(image)
    '''
        
    PlotMaps(image, estimator)
        
    # Disable the grid
    plt.grid(False)
    
    plt.show()
    
def RecognizeVideo(originalPath, destinationPath, clipped = False, beginTime = 0, endTime = 30):
    
    # Set the target window size
    width = 375
    height = 667
    
    name = "cmu"
    model = get_graph_path(name)
    
    # Instantiate the TfPoseEstimator
    estimator = TfPoseEstimator(model, target_size = (width, height))
    
    if clipped:
    
        # Initialize an instance of Video File Clip, trim from beginTime ~ endTime
        videoFileClip = VideoFileClip(originalPath).subclip(beginTime, endTime)
        
        # Process each frame
        whiteClip = videoFileClip.fl_image(lambda image: ProcessFrame(image, estimator))
        
        whiteClip.write_videofile(destinationPath)
        
    else:
        
        # Initialize an instance of Video File Clip
        videoFileClip = VideoFileClip(originalPath)
        
        # Process each frame
        whiteClip = videoFileClip.fl_image(lambda image: ProcessFrame(image, estimator))
        
        whiteClip.write_videofile(destinationPath)
    
def ProcessFrame(frame, estimator: TfPoseEstimator):
    
    # Infer the frame
    keyParts = estimator.inference(frame, resize_to_default = False, upsample_size = 4.0)
    # Draw the pose onto original frame
    image = TfPoseEstimator.draw_humans(frame, keyParts, imgcopy = False)
    
    return image

def PlotMaps(image, estimator):
    
    # Plot the paf map and heat map
    
    # 1st image
    figure = plt.figure()
    
    subplot = figure.add_subplot(2, 2, 1)
    
    # Set the title
    subplot.set_title("Result")
    
    # image is read by InferImages()
    # Convert the BGR t RGB to show the image
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.imshow(image)
    
    # Hide the grid
    plt.grid(False)
    
    # Show the colorful bar on the right
    plt.colorbar()
    
    # 2nd image
    backgroundImage = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
    
    # Reset the size of converted image, interplolation is to resampe
    backgroundImage = cv2.resize(backgroundImage, (estimator.heatMat.shape[1], estimator.heatMat.shape[0]), interpolation = cv2.INTER_AREA)
    
    subplot = figure.add_subplot(2, 2, 2)
    
    # Show the translucent image which controlled by alpha, the parameter is ranged between 0 and 1. 1 stands for opaque, and 0 transparent totally
    plt.imshow(backgroundImage, alpha = 0.5)
    
    # Invert the heatmat array alongside axis 2, obtain the maximum, that is the detected protruding points
    heatMat = np.amax(estimator.heatMat[:, :, : -1], axis = 2)
    
    # Show the image with gray and translucent style
    plt.imshow(heatMat, cmap = plt.cm.gray, alpha = 0.5)
    
    # Set the title for subplot
    subplot.set_title("Dot Network")
    
    plt.grid(False)
    
    plt.colorbar()
    
    # 3rd image
    # Transpose the pafMat array
    pafMat = estimator.pafMat.transpose((2, 0, 1))
    
    # Obtain the odd maximum from array alongside the axis 0
    oddMaximum = np.amax(np.absolute(pafMat[::2, :, :]), axis = 0)
    # Obtain the even maximum from array alongsie the axis 0
    evenMaximum = np.amax(np.absolute(pafMat[1::2, :, :]), axis = 0)
    
    subplot = figure.add_subplot(2, 2, 3)
    
    # Set the title
    subplot.set_title("Vector Map - Axis X")
    
    # Show the odd image with gray, translucent style
    plt.imshow(oddMaximum, cmap = plt.cm.gray, alpha = 0.5)
    
    plt.grid(False)
    plt.colorbar()
    
    # 4th image
    
    subplot = figure.add_subplot(2, 2, 4)
    
    # Set the title
    subplot.set_title("Vector Map - Axis Y")
    
    # Show the even image with gray, translucent style
    plt.imshow(evenMaximum, cmap = plt.cm.gray, alpha = 0.5)
    
    plt.colorbar()
    plt.grid(False)
    
    plt.show()

'''

files = glob.glob("../Inventory/Images/*.jpg")
InferImages(files[0])

'''

RecognizeVideo("../Inventory/Videos/test_video_1.mp4", "../Inventory/Videos/test_video_1_edited.mp4")