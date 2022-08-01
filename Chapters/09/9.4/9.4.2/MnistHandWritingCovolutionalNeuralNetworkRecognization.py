import tensorflow as tf
import keras
from keras.datasets import mnist
import math
from datasets import MNISTDataset

# Width and hight of the image
imageSize = 28 * 28
# 10 classifications for the image, digit 0 - 9
classesNumber = 10
# Learning Rate, 1e-4 is the scientific notation, value 0.0001
learningRate = 1e-4
# Number of Iteration
epochs = 10
# Size of batch
batchSize = 50

def Prepare():
    
    # Download and read the MNIST datasets
    (imagesTrain, labelsTrain), (imagesTest, labelsTest) = mnist.load_data()
    
    # Split 5000 for validation set
    validationLength = 5000
    imagesLength = imagesTrain.shape[0]
    
    trainLength = imagesLength - validationLength
    
    # Validation Set
    imagesValidation = imagesTrain[trainLength:]
    labelsValidation = labelsTrain[trainLength:]
    
    # Train Set
    imagesTrain = imagesTrain[: trainLength]
    labelsTrain = labelsTrain[: trainLength]
    
    # Convert the Train Set, Validation Set, Test Set to be vectorized image
    imagesTrain = imagesTrain.reshape(imagesTrain.shape[0], imageSize)
    imagesValidation = imagesValidation.reshape(imagesValidation.shape[0], imageSize)
    imagesTest = imagesTest.reshape(imagesTest.shape[0], imageSize)
    
    # Convert the Train Set, Validation Set and Test Set to float32
    imagesTrain = imagesTrain.astype("float32")
    imagesValidation = imagesValidation.astype("float32")
    imagesTest = imagesTest.astype("float32")
    
    # Convert the Train Set, Validation Set, Test Set to be the value between 0 - 1, normalization
    imagesTrain /= 255
    imagesValidation /= 255
    imagesTest /= 255
    
    # One Hoted the labels of Train Set, Validation Set and Test Set by to_categorical()
    labelsTrain = keras.utils.to_categorical(labelsTrain, classesNumber)
    labelsValidation = keras.utils.to_categorical(labelsValidation, classesNumber)
    labelsTest = keras.utils.to_categorical(labelsTest, classesNumber)
    
    # Show the shape of sets
    print("Shape of Train Images: {}".format(imagesTrain.shape))
    print("Shape of Train Labels: {}".format(labelsTrain.shape))
    print("Shape of Validation Images: {}".format(imagesValidation.shape))
    print("Shape of Validation Labels: {}".format(labelsValidation.shape))
    print("Shape of Test Images: {}".format(imagesTest.shape))
    print("Shape of Test Labels: {}".format(labelsTest.shape))
    
    return imagesTrain, labelsTrain, imagesValidation, labelsValidation, imagesTest, labelsTest
    

def CreateCovolutionalNeuralNetwork(inputData, inputChannelsNumber, filtersNumber, filterShape, poolShape, name):
    
    # The shape of filter of CNN is [filter height, filter width, in channels, out channels]
    filterShape = [filterShape[0], filterShape[1], inputChannelsNumber, filtersNumber]
    
    # The weight of Tensor variable. The initial value is Truncated Normal Distribution, Standard Deviation is 0.03
    weights = tf.Variable(tf.compat.v1.truncated_normal(filterShape, stddev = 0.03), name = name + "_W")
    
    # The bias of Tensor Variable. The initial value is Truncated Normal Distribution.
    bias = tf.Variable(tf.compat.v1.truncated_normal([filtersNumber]), name = name + "_b")
    
    # Define the Covolutional Neural Network
    # Parameter 1: Input Image
    # Parameter 2: Weights
    # Parameter 3: Step
    # Parameter 4: Padding. If padding = SAME means in a motion window, the data is less than filter will be padded by 0.
    # In the case padding = VALID, the data which is less than filter will be abandoned
    outLayer = tf.nn.conv2d(inputData, weights, (1, 1, 1, 1), padding = "SAME")
    outLayer += bias
    
    # Compute the output by activation method ReLU
    outLayer = tf.nn.relu(outLayer)
    
    # Add the Maximum Pooling layer, the shape of ksize is [batch, height, width, channels],
    # The shape of strides is [batch, stride, stride, channels]
    
    outLayer = tf.nn.max_pool(outLayer, ksize = (1, poolShape[0], poolShape[1], 1), strides = (1, 2, 2, 1), padding = "SAME")
    
    return outLayer

def AddCovolutionalNeuralNetworkLayer():
    
     # Define tbe input placeholder, the size of input image shape is [batch, image_vector]
    images = tf.compat.v1.placeholder(tf.float32, shaple = [None, imageSize])
    # The shape of input image on CNN is [batch, height, width, channels]
    shapedImages = tf.reshape(images, [-1, 28, 28, 1])
    
    # Define placeholder of the output
    labels = tf.compat.v1.placeholder(tf.float32, shape = [None, classesNumber])
    
    # Add the layer 1, depth is 32
    layer1 = CreateCovolutionalNeuralNetwork(shaped, 1, 32, (5, 5), (2, 2), name = "layer1")
    
    # Add the layer 2, depth is 64
    layer2 = CreateCovolutionalNeuralNetwork(layer1, 32, 64, (5, 5), (2, 2), name = "layer2")
    
    # Add the flattened layer
    flattened = tf.reshape(layer2, (-1, 7 * 7 * 64))
    
    # Add the full-connected layer
    wd1 = tf.Variable(tf.compat.v1.truncated_normal((7 * 7 * 64, 1000), stddev = 0.03), name = "wd1")
    bd1 = tf.Variable(tf.compat.v1.truncated_normal([1000], stddev = 0.01), name = "bd1")
    
    denseLayer1 = tf.add(tf.matmul(flattened, wd1), bd1)
    denseLayer2 = tf.nn.relu(denseLayer1)
    
    # Add the output full-connected layer, depth is 10 because there is only 10 classifications
    wd2 = tf.Variable(tf.compat.v1.truncated_normal((1000, classesNumber), stddev = 0.03), name = "wd2")
    bd2 = tf.Variable(tf.compat.v1.truncated_normal([classesNumber], stddev = 0.01), name = "db2")
    
    denseLayer2 = tf.add(tf.matmul(denseLayer1, wd2), bd2)
    
    # Add the activation method softmax output layer
    labels_ = tf.nn.softmax(denseLayer2)
    
    # Compute the loss via softmax crossentropy
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = labels_, labels = labels))
    
    # Define the optimizer to be adam
    optimizer = tf.train.AdamOptimier(learning_rate = learningRate).minimize(cost)
    
    # Compare the correct prediction
    correctPrediction = tf.equal(tf.argmax(labels, 1), tf.argmax(labels_, 1))
    
    # Compute the accuracy
    accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))
    
def Train(imagesTrain, labelsTrain, imagesTest, labelsTest):
    
    # Create session of TensorFlow
    with tf.compat.v1.Session() as session:
        
        # Initialize global variable of tensorflow
        session.run(tf.compat.v1.global_variables_initializer())
        
        # Compute the numbers of training for all train sets when batch equals batchSize
        batchCount = int(math.ceil(imagesTrain.shape[0] / float(batchSize)))
        data = MNISTDataset(imagesTrain.reshape([-1, 784]), labelsTrain, imagesTest.reshape([-1, 784]), labelsTest, batch_size = batchSize)
        
        # Iterate number of epochs
        for e in range(epochs):
            
            # Train each image
            for i in range(batchCount):
                
                # Grab batch_size images each time
                batchImages = data.next_batch(batchSize)
                
                print(batchImages)
        
        # The Saver of model persistence
        v1 = tf.Variable(5, name = "v1")
        v2 = tf.Variable(6, name = "v2")
        
        saver = tf.compat.v1.train.Saver([v1, v2])
        saver.save(session, "checkpoint/mnist_cnn_tf.ckpt")
                

imagesTrain, labelsTrain, imagesValidation, labelsValidation, imagesTest, labelsTest = Prepare()

Train(imagesTrain, labelsTrain , imagesTest, labelsTest)