import os
import gzip
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow.compat.v1 as tf
import math

tf.disable_v2_behavior()

import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image
from glob import glob
from matplotlib import image
from PIL import Image, ImageOps
from MnistFaceGenerativeAdversarialNetworksDataset import MnistFaceGenerativeAdversarialNetworksDataset

def ReadStream(bytesStream):
    
    '''

    Read 32bit integers from bytes stream
    
    Parameter 0 bytesStream: Bytes Stream to read
    
    Return: 32-bit Integers
    
    '''
    
    dateType = np.dtype(np.uint32).newbyteorder('>')
    
    return np.frombuffer(bytesStream.read(4), dtype = dateType)[0]

def Decompress(gzFilePath):
    
    '''

    Decompress the gz file, and convert the array into files
    
    Parameter 0 gzFilePath: path of the gz file
    Parameter 1 storePath: where to store the extracted files
    
    '''
    
    # Read the MNIST dataset directly
    with open(gzFilePath, "rb") as f:
        
        # gzip is single file/lossless data compression utility, the extension name for the compressed file is generally gz
        # Read the dataset into bytes stream
        with gzip.GzipFile(fileobj = f) as byteStream:
            
            # Handle the bytes stream
            magic = ReadStream(byteStream)
            
            # 2051 represents magic number which is decribed in manual. Magic number descriminates the images and labels
            # 2051 represents the images, 2049 represents labels
            
            if magic != 2051:
                
                raise ValueError("Invalid magic number in file: {}".format(magic, f.name))
            
            imagesNumber = ReadStream(byteStream)
            
            rows = ReadStream(byteStream)
            columns = ReadStream(byteStream)
            
            buffer = byteStream.read(rows * columns * imagesNumber)
            
            array = np.frombuffer(buffer, dtype = np.uint8)
            array = array.reshape(imagesNumber, rows, columns)
            
            return array
        
def StoreImages(array, path):
    
    # Iterate and store the gray images
    for i, image in enumerate(tqdm(array, unit = "File", unit_scale = True, miniters = 1, desc = "Extract the images from MNIST dataset")):
        
        storePath = os.path.join(path, "Image_{}.jpg".format(i))
        
        # Save the image in the specific path
        Image.fromarray(image, "L").save(storePath)
        
def ExtractImages(gzFilePath, storePath):
    
    array = Decompress(gzFilePath)
    
    StoreImages(array, storePath)
    
def PlotRandomImages(paths, quatity):
    
    # Eandomly select length of [k] unique images from entire sequence
    # Select number of [quanlity] paths of images
    
    randomImages = random.sample(paths, quatity)
    
    # Create the object of plot, 5 rows, 5 columns
    figure, axes = plt.subplots(nrows = 5, ncols = 5)
    
    # Set the size of figure
    figure.set_size_inches(8, 8)
    
    i = 0
    
    # Row
    for row in range(5):
        # Column
        for column in range(5):
            
            _image = image.imread(randomImages[i])
            
            # Acquire the obejct of axes accroding to [row, column]
            ax = axes[row, column]
            
            # Show the image on ax
            ax.imshow(_image)
            
            # Hide the grid
            ax.grid(False)
            
            i += 1
            
    plt.show()
    
def CreateInputs(width, height, channels, zDimension):
    
    '''

    Create the input placeholders for model
    
    Parameter 0 width: the width of input image
    Parameter 1 height: the height of input image
    Parameter 2 channels: the number of channels of input image
    Parameter 3 zDimension: Z dimension
    
    Return: Tuple of (tensor of real input images, tensor of z data, leanring rate
    
    
    
    inputReal = tf.keras.Input(name = "inputReal", shape = (None, width, height, channels), dtype = tf.dtypes.float32)
    inputZ = tf.keras.Input(name = "inputZ", shape = (None, zDimension), dtype = tf.dtypes.float32)
    learningRate = tf.keras.Input(name = "learningRate", shape = (), dtype = tf.dtypes.float32)
    
    return inputReal, inputZ, learningRate
    
    '''
    inputReal = tf.placeholder(tf.float32, [None, width, height, channels], name = "inputReal")
    inputZ = tf.placeholder(tf.float32, [None, zDimension], name = "inputZ")
    learningRate = tf.placeholder(tf.float32, name = "learningRate")
    
    return inputReal, inputZ, learningRate
    

def EnableLeakyReLU(value, alpha):
    
    return tf.maximum(value * alpha, value)

def CreateDiscriminator(images, reuse = False):
    
    '''

    Create a Discriminator Network
    
    Parameter 0 images: the tensor of input images
    Parameter 1 reuse: whether the weight is going to be reused
    
    Return: Tensor of the Discriminator output, the tensor of logits of Discriminator
    
    '''
    
    # Use tf.initializers.glorot_uniform() as the kernel_initializerparameter, this will accelerate the convergence of train
    
    alpha = 1
    
    with tf.variable_scope("discriminator", reuse = reuse):
        
        conv1 = tf.layers.conv2d(images, 64, 5, strides = 2, padding = "same", kernel_initializer = tf.initializers.glorot_uniform()),
        # Leaky ReLU
        conv1 = tf.maximum(alpha * conv1, conv1)
        conv1 = tf.nn.dropout(conv1, 0.9)
        
        conv2 = tf.layers.conv2d(conv1, 128, 5, strides = 2, padding = "same", kernel_initializer = tf.initializers.glorot_uniform())
        conv2 = tf.layers.batch_normalization(conv2, training = True)
        # Leaky ReLU
        conv2 = tf.maximum(alpha * conv2, conv2)
        conv2 = tf.nn.dropout(conv2, 0.9)
        
        conv3 = tf.layers.conv2d(conv2, 256, 5, strides = 2, padding = "same", kernel_initializer = tf.initializers.glorot_uniform())
        conv3 = tf.layers.batch_normalization(conv3, training = True)
        # Leaky ReLU
        conv3 = tf.maximum(alpha * conv3, conv3)
        conv3 = tf.nn.dropout(conv3, 0.9)
        
        flat = tf.reshape(conv3, (-1, 4 * 4 * 256))
        logits = tf.layers.dense(flat, 1)
        
        output = tf.sigmoid(logits)
        
        return output, logits
        
def CreateGenerator(z, outputChannelDimensions, isTrain = True, resue = True):
    
    '''

    Create a Generator Network
    
    Parameter 0 z: input z
    Parameter 2 outputChannelDimensions: The number of output channels
    Parameter 3 isTrain: Whether the train is needed for the Generator
    Parameter 4 resue: Whether to reuse the Generator
    
    Return: Tensor of the Generator
    
    '''
    
    alpha = 1
    
    with tf.variable_scope("generator", reuse = not isTrain):
        
        dense1 = tf.layers.dense(z, 7 * 7 * 512)
        dense1 = tf.reshape(dense1, (-1, 7, 7, 512))
        dense1 = tf.layers.batch_normalization(dense1, training = isTrain)
        dense1 = tf.maximum(alpha * dense1, dense1)
        dense1 = tf.nn.dropout(dense1, 0.5)
        
        conv1 = tf.layers.conv2d_transpose(dense1, 256, 5, strides = 2, padding = "same")
        conv1 = tf.layers.batch_normalization(conv1, training = isTrain)
        conv1 = tf.maximum(alpha * conv1, conv1)
        conv1 = tf.nn.dropout(conv1, 0.5)
        
        conv2 = tf.layers.conv2d_transpose(conv1, 128, 5, strides = 2, padding = "same")
        conv2 = tf.layers.batch_normalization(conv2, training = isTrain)
        conv2 = tf.maximum(alpha * conv2, conv2)
        conv2 = tf.nn.dropout(conv2, 0.5)
        
        logits = tf.layers.conv2d_transpose(conv2, outputChannelDimensions, 5, strides = 1, padding = "same")
        
        output = tf.tanh(logits)
        
        return output
    
def ComputeLoss(inputReal, inputZ, outputChannelDimensions):
    
    '''

    Acquire the loss of Discriminator and Generator
    
    Parameter 0 inputReal: Input of real image
    Parameter 1 inputZ: Input of Z
    Parameter 2 outputChannelDimensions: Number of channels of output image
    
    Return: Tuple of (Loss of Discrimintor, Loss of Generator)
    
    '''
    
    smooth = 0.1
    
    # Build the model of Generator
    generatorModel = CreateGenerator(inputZ, outputChannelDimensions)
    
    # Build the model of Discriminator
    realDiscriminatorModel, realDiscriminatorLogits = CreateDiscriminator(inputReal)
    fakeDiscriminatorModel, fakeDiscriminatorLogits = CreateDiscriminator(generatorModel, reuse = True)
    
    # Compute the losses of disciminators for real image and fake image
    realLoss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = realDiscriminatorLogits, labels = tf.ones_like(realDiscriminatorModel) * (1 - smooth)))
    fakeLoss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = fakeDiscriminatorLogits, labels = tf.zeros_like(fakeDiscriminatorModel)))
    
    discriminatorLoss = realLoss + fakeLoss
    
    # Compute the loss of Generator
    generatorLoss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = fakeDiscriminatorLogits, labels = tf.ones_like(fakeDiscriminatorModel)))
    
    return discriminatorLoss, generatorLoss

def CreateOptimizer(discriminatorLoss, generatorLoss, learningRate, beta1):
    
    '''

    Acquire the optimizer
    
    Parameter 0 discriminatorLoss: the tensor of discriminator loss
    Parameter 1 generatorLoss: the tensor of generator loss
    Parameter 2 learningRate: percentage of learning
    Parameter 3 beta1: exponential decay rate of 1st time in Optimizer
    
    Return: Tuple of (Train operation of Discriminator, Train operation of Generator)
    
    '''
    
    #  Acquire the weights and deviation for the purpose of updating
    
    trainableVariables = tf.trainable_variables()
        
    discriminatorVariables = [var for var in trainableVariables if var.name.startswith("discriminator")]
    generatorVariables = [var for var in trainableVariables if var.name.startswith("generator")]
    
    # Build the Optimizer
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        
        discriminatorTrainOptimizer = tf.train.AdamOptimizer(learningRate, beta1 = beta1).minimize(discriminatorLoss, var_list = discriminatorVariables)
        generatorTrainOptimizer = tf.train.AdamOptimizer(learningRate, beta1 = beta1).minimize(generatorLoss, var_list = generatorVariables)
        
        return discriminatorTrainOptimizer, generatorTrainOptimizer

def SquareImage(images, mode):
    
    '''

    Save the images with squred grid
    
    Parameter 0 images: the image to be saved with square grid
    Parameter 1 mode: mode of the image, gray or RGB
    
    Return: An sqaured image object of Image type
    
    '''
    
    # Acquire the maximum size of squared grid image
    size = int(math.floor(np.sqrt(images.shape[0])))
    
    # Change the values in image, make the value be between 0 to 255
    images = (((images - images.min()) * 255) / (images.max() - images.min())).astype(np.uint8)
    
    # Place the images in a square
    squaredImages = np.reshape(images[: size * size], (size, size, images.shape[1], images.shape[2], images.shape[3]))
    
    # Image is gray if model == L
    if mode == "L":
        
        squaredImages = np.squeeze(squaredImages, 4)
        
    # Create new grid image with combination of images
    newImage = Image.new(mode, (images.shape[1] * size, images.shape[2] * size))
    
    for i, sis in enumerate(squaredImages):
        
        for j, image in enumerate(sis):
            
            array = Image.fromarray(image, mode)
            
            if mode == "L":
                
                array = ImageOps.invert(array)
                
            newImage.paste(array, (i * images.shape[1], j * images.shape[2]))
            
    return newImage

def ShowGeneratorOutput(sess, imagesNumber, inputZ, outputChannelDimensions, mode):
    
    '''

    Show the sampling image outputed by Generator
    
    Parameter 0 sess: Session of TensorFlow
    Parameter 1 imagesNumber: Number of images to show
    Parameter 2 inputZ: Z tensor of input
    Parameter 3 outputChannelDimensions: Number of output channels
    Parameter 4 mode: mode the image uses, L or RGB
    
    '''
    
    cmap = None if mode == "RGB" else "gray"
    
    zDimensions = inputZ.get_shape().as_list()[-1]
    exampleZ = np.random.uniform(-1, 1, size = [imagesNumber, zDimensions])
    
    # Sampling after generated by Generator
    samples = sess.run(
        CreateGenerator(inputZ, outputChannelDimensions, False),
        feed_dict = {inputZ: exampleZ})
    
    # Convert the data of image grid
    
    imageGrid = SquareImage(samples, mode)
    
    plt.imshow(imageGrid, cmap = cmap)
    plt.grid(False)
    
    plt.show()
    
def Train(epochs, batchSize, zDimensions, learningRate, beta1, functionGetBatches, shape, mode):
    
    '''

    Train and generate the model of Generative Adversarial Network
    
    Parameter 0 epoches: Number of epoches
    Parameter 1 batchSize: Size of batch
    Parameter 2 zDimensions: Z Dimensions
    Parameter 3 learningRate: Learning Rate
    Parameter 4 beta1: Exponential decay rate of 1st time in Optimizer
    Parameter 5 functionGetBatches: Function to Get Batches
    Parameter 6 shape: Shape of data
    Parameter 7 mode: mode of image, L or RGB
    
    '''
    
    # Acquire the tensor of input model
    inputReal, inputZ, learingRate = CreateInputs(shape[1], shape[2], shape[3], zDimensions)
    
    # Acquire the loss of model
    discriminatorLoss, generatorLoss = ComputeLoss(inputReal, inputZ, shape[3])
    
    # Acquire the optimizer of model
    discriminatorOptimizer, generatorOptimizer = CreateOptimizer(discriminatorLoss, generatorLoss, learningRate, beta1)
    
    # Start the session
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        
        # Start the train iterations
        iteration = 0
        
        for i in range(epochs):
            
            batchesGenerator = functionGetBatches(batchSize)
            
            for batchImages in batchesGenerator:
                
                # Begin train
                # Generate an images matrix randomly
                z_ = np.random.uniform(-1, 1, (batchSize, zDimensions))
                
                # Update the discriminator
                _ = sess.run(discriminatorOptimizer, feed_dict = {inputReal: batchImages * 2, inputZ: z_})
                
                # Update the generator
                _ = sess.run(generatorOptimizer, feed_dict = {inputZ: z_, inputReal: batchImages, })
                
                iteration += 1
                
                # Print the logs each 10 iterations
                if iteration % 10 == 0:
                    
                    discriminatorLoss_ = discriminatorLoss.eval({inputZ: z_, inputReal: batchImages})
                    generatorLoss_ = generatorLoss.eval({inputZ: z_})
                    
                    print("Iteration: {}, discriminatorLoss_ = {:.5f}, generatorLoss_ = {:.5f}".format(iteration, discriminatorLoss_, generatorLoss_))
                    
                # Preview the image each 50 iterations in order to check the effect
                if iteration % 50 == 0:
                    
                    ShowGeneratorOutput(sess, 25, inputZ, shape[3], mode)
    
def Start():
    
    base = "../Exclusion/"
    extraction = base + "Extraction/"
    
    paths = glob(extraction + "*.jpg")
    
    if len(paths) == 0:
        
        # Path of MNIST dataset
        ExtractImages(base + "train-images-idx3-ubyte.gz", extraction)
        
    print(paths[: 10])
    print(len(paths))
    
    PlotRandomImages(paths, 25)

    # Batch Size
    batchSize = 64
    # Input Z dimensions
    zDimensions = 100
    # Learing Rate
    learningRate = 0.001
    # Exponential decay rate
    beta1 = 0.5
    # Number of epochs
    epochs = 20
    
    # Initialize MNIST dataset via the paths of all MNIST images
    mnistDataset = MnistFaceGenerativeAdversarialNetworksDataset("mnist", paths)
    
    with tf.Graph().as_default():
        
        Train(epochs, batchSize, zDimensions, learningRate, beta1, mnistDataset.GetBatches, mnistDataset.shape, mnistDataset.mode)
    
Start()