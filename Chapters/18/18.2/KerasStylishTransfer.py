import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

from PIL import Image
from keras import backend
from keras.applications.vgg16 import VGG16
from scipy.optimize import fmin_l_bfgs_b

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
        
        layerFeatures = layers[name]
        
        styleFeatures = layerFeatures[1, :, :, :]
        combinationFeatures = layerFeatures[2, :, :, :]
        
        styleLoss = ComputeStyleLoss(styleFeatures, combinationFeatures)
        
        loss = loss + (styleWeights / len(featureLayers)) * styleLoss
        
    layerFeatures = layers["block2_conv2"]
    
    contentImageFeatures = layerFeatures[0, :, :, :]
    combinationFeatures = layerFeatures[2, :, :, :]
    
    loss += contentWeights * ComputeContentLoss(contentImageFeatures, combinationFeatures)
    loss += totalVariationWeights * ComputeVariationLoss(combinationImage)
    
    # Compute the gradient
    gradient = backend.gradients(loss, combinationImage)
    
    '''
    with tf.GradientTape() as tape:
        
        gradient = tape.gradient(loss, combinationImage)
    '''
    
    outputs = [loss]
    
    if isinstance(gradient, (list, tuple)):
        
        outputs += gradient
        
    else:
        
        outputs.append(gradient)
        
    # Initialize the function of keras
    outputsFunction = backend.function([combinationImage], outputs)
    
    return outputsFunction
    
def EvaluateLossGradient(array, function):
    
    array = array.reshape((1, height, width, 3))
    
    outputs = function([array])
    
    lossValue = outputs[0]
    gradientValue = outputs[1].flatten().astype("float64")
    
    return lossValue, gradientValue

class Evaluator(object):
    
    def __init__(self, function):
        
        self.lossValue = None
        self.gradientValue = None
        
        self.function = function
        
    # Compuete the loss
    def CompueteLoss(self, array):
        
        assert self.lossValue is None
        
        lossValue, gradientValue = EvaluateLossGradient(array, self.function)
        
        self.lossValue = lossValue
        self.gradientValue = gradientValue
       
        return self.lossValue
    
    def CompueteGradient(self, array):
        
        assert self.lossValue is not None
        
        gradientValue = np.copy(self.gradientValue)
        
        self.lossValue = None
        self.gradientValue = None
        
        return gradientValue
    
def ComputeContentLoss(content, combination):
    
    return backend.sum(backend.square(content - combination))

def GramMatrix(array):
    
    # Transpose the image array, and flatten
    features = backend.batch_flatten(backend.permute_dimensions(array, (2, 0, 1)))
    
    # Return the dot computing
    gram = backend.dot(features, backend.transpose(features))
    
    return gram

# Compuete the Style Loss
def ComputeStyleLoss(style, combination):
    
    # Compute the matrix of gram for style
    styleGram = GramMatrix(style)
    
    # Compuete the matrix ofgram for combination
    combinationGram = GramMatrix(combination)
    
    channels = 3
    size = height * width
    
    loss = backend.sum(backend.square(styleGram - combinationGram)) / (4 * (channels ** 2) * (size ** 2))
    
    return loss

def ComputeVariationLoss(array):
    
    a = backend.square(array[:, : height - 1, : width - 1, :] - array[:, 1:, : width - 1, :])
    b = backend.square(array[:, : height - 1, : width - 1, :] - array[:, : height - 1, 1:, :])
    
    return backend.sum(backend.pow(a + b, 1.25))

def Generate(function):
    
    image = np.random.uniform(0, 255, (1, height, width, 3)) - 128.0
    
    evaluator = Evaluator(function)
        
    iterations = 10
    
    for i in range(iterations):
        
        print("Iteration {}".format(i + 1))
        
        # Begin clocking
        beginTime = time.time()
        
        # Parameter 0: Expected minimized loss function
        # Parameter 1: An initialized NumPy array
        # Parameter 2: Function to compute gradient
        # Parameter 3: Maximum function to evaliuate
        array, minimum, info = fmin_l_bfgs_b(evaluator.CompueteLoss, x.flatten(), fprime = evaluator.CompueteGradient, maxfun = 20)
        
        print("Minimum = {}".format(minimum))
        
        # End clocking ont time
        endTime = time.time()
        
        print("Iteration {} is completed in {:.2f}s".format(i, endTime - beginTime))
        
    return array

def ProcessImage(array):
    
    array = array.reshape((height, width, 3))
    array = array[:, :, ::-1]
    
    array[:, :, 0] += 103.939
    array[:, :, 1] += 116.779
    array[:, :, 2] += 123.68
    
    array = np.clip(array, 0, 255).astype("uint8")
    
    return array

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
    
    function = CreateModel(contentImageArray, styleImageArray)
    
    array = Generate(function)
    array = ProcessImage(array)
    
    Image.fromarray(array)

Start()
