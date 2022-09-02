import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from keras import backend
from keras.applications.vgg16 import VGG16

# Size of image
width = 512
height = 512

def LoadImage(imagePath):
    
     # Load the image
    image = Image.open(imagePath)
    
    lanczos = Image.Resampling.LANCZOS
    
    # Resize the image
    image = image.resize((width, height))
    
    # convert the values in image into float32
    array = np.asarray(image, dtype = np.float32)
    
    # Expand the dimensions to batch dimensions, which is expected shape by RNN
    array = np.expand_dims(array, axis = 0)
        
    return array

def PresentImage(array, title):
    
    # Remove the batch dimension
    image = np.squeeze(array.astype("uint8"), axis = 0)
    
    plt.grid(False)
    
    plt.title(title)
    plt.imshow(image)
    
    plt.show()

def Reverse(processedImage):
    
    # Perform the steps of deprocession. For the normalization, 3 channels respectively are,
    processedImage[:, :, :, 0] += 103.939
    processedImage[:, :, :, 1] += 116.779
    processedImage[:, :, :, 2] += 123.68
    
    # Reverse
    array = processedImage[:, :, :, ::-1]
    
    return array

def CreateModel(contentImageArray, styleImageArray):
    
    # Create the content and style variables of tensorflow
    contentImage = backend.variable(contentImageArray)
    styleImage = backend.variable(styleImageArray)
    
    # Input tensor
    combinationImage = backend.placeholder((1, height, width, 3))
    inputTensor = backend.concatenate([contentImage, styleImage, combinationImage], axis = 0)
    
    # Global loss
    loss = backend.variable(0.)
    
    # Content weights
    contentWeights = 0.05
    
    # Style weights
    styleWeights = 5.0
    
    # Global variable weights
    totalVariationWeights = 1.0
    
    # Obtain the model fo VGG16. Model will be downloaded in the case the model does not exist
    model = VGG16(input_tensor = inputTensor, weights = "imagenet", include_top = False)
    
    # Acquire all the layer in this model
    layers = dict([(layer.name, layer.output) for layer in model.layers])
    
    # All the needed convolutional layers in the model
    featureLayers = ["block1_conv2", "block2_conv2", "block3_conv3", "block4_conv3", "block5_conv3"]
    
    # Compute the loss for each layer
    for name in featureLayers:
        
        layerFeatures = layer[name]
        styleFeatures = layerFeatures[1, :, :, :]
        combinationFeatures = layerFeatures[2, :, :, :]
        
        styleLoss = ComputeStyleLoss(styleFeatures, combinationFeatures)
        
        loss += (styleWeights / len(featureLayers)) * styleLoss
        
    
        
def ComputeStyleLoss(style, combination):
    
    print("aaa")
    
def ComputeContentLoss(content, combination):
    
    return backend.sum(backend.square(content - combination))
    
def Start():
    
    # Content Image
    contentPath = "../Exclusion/Images/xiangshan.jpeg"
    
    plt.figure(figsize = (18, 9))
    
    # Draw the content image
    plt.subplot(1, 2, 1)
    
    contentImage = LoadImage(contentPath)
    PresentImage(contentImage, "Content Image")
    
    # Style Image
    stylePath = "../Exclusion/Images/the_shipwreck_of_the_minotaur.jpg"
    
    plt.figure(figsize = (18, 9))
    
    plt.subplot(1, 2, 2)
    
    styleImage = LoadImage(stylePath)
    PresentImage(styleImage, "Stylish Image")
    
    contentImageArray = Reverse(contentImage)
    styleImageArray = Reverse(styleImage)
    
    print("contentImageArray.shape = {}, styleImageArray.shape = {}".format(contentImageArray.shape, styleImageArray.shape))

Start()
