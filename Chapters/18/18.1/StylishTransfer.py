import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["figure.figsize"] = (12, 15)
mpl.rcParams["axes.grid"] = False

import numpy as np
import time
import tensorflow as tf

from PIL import Image
from tensorflow.keras.utils import img_to_array
from tensorflow.python.keras import models

# Content layer name of features mapping layer
contentLayers = ["block5_conv2"]
    
# Names of stylized layers
stylishLayers = ["block1_conv1",
                "block2_conv1",
                "block3_conv1",
                "block4_conv1",
                "block5_conv1"]
    
def LoadImage(imagePath):
    
    # Load the image
    image = Image.open(imagePath)
    
    # Get the maximum from width and heigh
    long = max(image.size)
    
    # Change the long of image to 512
    maximumDimension = 512
    scale = maximumDimension / long
    
    image = image.resize((round(image.size[0] * scale), round(image.size[1] * scale)), Image.ANTIALIAS)
    
    # Preprocess the image, make it to be a 3 dimension array
    image = img_to_array(image)
    
    # Expand the dimensions to batch dimensions, which is expected shape by RNN
    image = np.expand_dims(image, axis = 0)
        
    return image

def PresentImage(image, title):
    
    # Remove the batch dimension
    image = np.squeeze(image, axis = 0)
    
    plt.title(title)
    plt.imshow(image)
    
    plt.show()
    
def Process(imagePath):
    
    # Load the image
    image = LoadImage(imagePath)
    
    # Process the image with VGG19
    image = tf.keras.applications.vgg19.preprocess_input(image)
    
    return image

def Reverse(processedImage):
    
    # Duplicate the image
    _image = processedImage.copy()
    
    # If the dimension is 4, remove the batch dimension
    if len(_image.shape) == 4:
        
        _image = np.squeeze(_image, 0)
        
    assert len(_image.shape) == 3, ("The dimensions of input processed image should be [1, height, width, channel] or [height, width, channel]")
    
    if len(_image.shape) != 3:
        
        raise ValueError("Invalid image")
    
    # Perform the steps of deprocession. For the normalization, 3 channels respectively are,
    _image[:, :, 0] += 103.939
    _image[:, :, 1] += 116.779
    _image[:, :, 2] += 123.68
    
    # Reverse
    _image = _image[:, :, ::-1]
    
    # Cut the values on array to 0 ~ 255
    _image = np.clip(_image, 0, 255).astype("uint8")
    
    return _image

def CreateModel():
    
    # Load the pretrained model VGG19, the model is trained by the images on imagenet
    # include_top is False means the fully-connected layer which VGG19 NN model top layer
    vgg = tf.keras.applications.vgg19.VGG19(include_top = False, weights = "imagenet")
    
    # Set the layers not to be trained
    vgg.trainable = False
    
    # Get the style associated output layer
    styleOutputs = [vgg.get_layer(name).output for name in stylishLayers]
    
    # Get the output layer corresponds to content layer
    contentOutputs = [vgg.get_layer(name).output for name in contentLayers]
    
    modelOutputs = styleOutputs + contentOutputs
    
    # Build the model
    return models.Model(vgg.input, modelOutputs)

def ComputeContentLoss(baseContent, target):
    
    return tf.reduce_mean(tf.square(baseContent - target))

def GramMatrix(inputTensor):
    
    # Image channel first
    channels = int(inputTensor.shape[-1])
    
    inputTensor1 = tf.reshape(inputTensor, [-1, channels])
    inputTensor2 = tf.shape(inputTensor1)[0]
    
    # Features mapping matmul
    gram = tf.matmul(inputTensor1, inputTensor1, transpose_a = True)
    
    return gram / tf.cast(inputTensor2, tf.float32)
    
def ComputeStyleLoss(baseStyle, gramTarget):
    
    # Each layer has height, width, gramTarget
    # Compute the loss on gievn layer via the size of features mapping and size of filter layers
    height, width, channels = baseStyle.get_shape().as_list()
    
    gramStyle = GramMatrix(baseStyle)
    
    # Compute the styple loss
    return tf.reduce_mean(tf.square(gramStyle - gramTarget))

def AcquireFeatureRepresentations(model, stylePath, contentPath):
    
    # Load image
    styleImage = Process(stylePath)
    contentImage = Process(contentPath)
    
    # Compute the features of style and content in batch
    styleOutputs = model(styleImage)
    contentOutputs = model(contentImage)
    
    # Acquire the stylish and content features from model
    styleFeatures = [styleLayer[0] for styleLayer in styleOutputs[: len(stylishLayers)]]
    contentFeatures = [contentLayer[0] for contentLayer in contentOutputs[len(stylishLayers):]]
    
    return styleFeatures, contentFeatures

def ComputeLoss(model, lossWeights, initialImage, gramStyleFeatures, contentFeatures):
    
    '''

    Compute the total loss
    
    Parameter 0 model: the medium layer is accessible to the model
    Parameter 1 lossWeights: each contribution weight to each loss function. The shape is [style, weights, content weights, style weights, total variation weights)
    Parameter 2 initialImage: Basic image for initialization which is used to optimize and compute the gradient loss
    Parameter 3 gramStyleFeatures: The gram matrix of precomputed and defined style layer
    Parameter 4 contentFeatures: Precomputed output from defined content layer
    
    '''
    
    styleWeights, contentWeights = lossWeights
    
    # Acquire the initial image via model, the initial image proovides the needed content and style
    modelOutputs = model(initialImage)
    
    # Get the output layer of style and content
    styleOutputFeatures = modelOutputs[: len(stylishLayers)]
    contentOutputFeatures = modelOutputs[len(stylishLayers):]
    
    styleScore = 0
    contentScore = 0
    
    # Accumlate the losses from all upper layers for style
    # Measure each contribution in each layer
    weightPerStyleLayer = 1.0 / float(len(stylishLayers))
    
    for targetStyle, combinedStyle in zip(gramStyleFeatures, styleOutputFeatures):
        
        styleScore += weightPerStyleLayer * ComputeStyleLoss(combinedStyle[0], targetStyle)
        
    # Accumlate the losses from all upper layers for content
    weightPerContentLayer = 1.0 / float(len(contentLayers))
    
    for targetContent, combinedContent in zip(contentFeatures, contentOutputFeatures):
        
        contentScore += weightPerContentLayer * ComputeContentLoss(combinedContent[0], targetContent)
        
    styleScore *= styleWeights
    contentScore *= contentWeights
    
    # Compute all the losses
    loss = styleScore + contentScore
    
    return loss, styleScore, contentScore

# Compute Gradient
def ComputeGradient(config):
    
    with tf.GradientTape() as tape:
        
        allLosses = ComputeLoss(**config)
        
    # Compute the gradience of input image
    totalLosses = allLosses[0]
    
    return tape.gradient(totalLosses, config["initialImage"]), allLosses
   
def TrainModel(contentPath, stylePath):
    
    # Don't need to train the whole VGG19
    model = CreateModel()
    
    for layer in model.layers:
        
        layer.trainable = False
        
    # Get the style and content features from specific medium layer
    styleFeatures, contentFeatures = AcquireFeatureRepresentations(model = model, stylePath = stylePath, contentPath = contentPath)
    gramStyleFeatures = [GramMatrix(feature) for feature in styleFeatures]
    
    # Set the initial image
    initialImage = Process(contentPath)
    initialImage = tf.Variable(initialImage, dtype = tf.float32)
    
    # Create the optimizer of Adam
    optimizer = tf.keras.optimizers.Adam(learning_rate = 5, beta_1 = 0.99, epsilon = 1e-1)
    
    # Store the result
    bestLoss, bestImage = float("inf"), None
    
    # Create basic configurations
    styleWeights = 1e-2
    contentWeights = 1e3
    
    lossWeights = (styleWeights, contentWeights)
    
    # model, lossWeights, initialImage, gramStyleFeatures, contentFeatures
    config = {"model": model, "lossWeights": lossWeights, "initialImage": initialImage, "gramStyleFeatures": gramStyleFeatures, "contentFeatures": contentFeatures}
    
    # Number of iterations
    iterationsNumber = 1000
    
    # Define a plot with 2 rows, 5 columns to display the stylized image with specific durations
    rows = 2
    columns = 5
    
    # Store the trained data when numer of trains elapse
    displayInterval = iterationsNumber / (rows * columns)
    
    beginTime = time.time()
    globalBeginTime = beginTime
    
    # Normal means
    normalMeans = np.array([103.939, 116.779, 123.68])
    
    minimum = -normalMeans
    maximim = 255 - normalMeans
    
    images = []
    
    # Start the trains
    for i in range(iterationsNumber):
        
        # Compute the gradient
        gradient, allLosses = ComputeGradient(config)
        loss, styleScore, contentScore = allLosses
        
        optimizer.apply_gradients([(gradient, initialImage)])
        
        # Cut and make the valeus scoped
        clipped = tf.clip_by_value(initialImage, minimum, maximim)
        
        initialImage.assign(clipped)
        
        endTime = time.time()
        
        if loss < bestLoss:
            
            # Update the best loss and image from all losses
            bestLoss = loss
            bestImage = Reverse(initialImage.numpy())
         
        print("Iteration: {}".format(i))
         
        # Store the trained data every 100 iterations
        if i % displayInterval == 0:
            
            beginTime = time.time()
            
            plotImage = initialImage.numpy()
            plotImage = Reverse(plotImage)
            
            images.append(plotImage)
            
            print("Iteration: {}".format(i))
            print("Total Loss: {: .4e}, Style Loss: {: .4e}, Content Loss: {: .4e}, Time: {: .4f}s".format(loss, styleScore, contentScore, time.time() - beginTime))
            print("Total Time: {: .4f}s".format(time.time() - globalBeginTime))
        
    # Plot the 10 images
    plt.figure(figsize = (14, 4))
        
    for i, image in enumerate(images):
            
        plt.subplot(rows, columns, i + 1)
        plt.imshow(image)
            
        plt.xticks([])
        plt.yticks([])
        
    Image.fromarray(bestImage)
    
    
def Start():
    
    # Content Image
    imagePath = "../Exclusion/Images/wangjing_selfie.jpg"
    
    # Draw the content image
    plt.subplot(1, 2, 1)
    
    image = LoadImage(imagePath).astype("uint8")
        
    PresentImage(image, "Content Image")
    
    # Style Image
    stylePath = "../Exclusion/Images/udnie.jpg"
    plt.subplot(1, 2, 2)
    
    image = LoadImage(stylePath).astype("uint8")
        
    PresentImage(image, "Stylish Image")
    
    TrainModel(imagePath, stylePath)
    
Start()
