import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import sys
sys.path.insert(1, "../../9.1/9.1.2/")
from MnistHandWritingConvolutionalPreparation import Prepare

def CreateModel(imagesTest, labelsTest, imagesValidation, labelsValidation, imageSize = 28 * 28, numberClasses = 10, learningRate = 0.1):
    
    # x here is the input. Create the input placeholder, the placeholder will fill out the iterated data when the placeholder is being trained 
    x = tf.compat.v1.placeholder(tf.float32, [None, imageSize])
    
    # W means the weight. Create the weight, and initilized it by 0, its size is (vector size of image, total categories of image
    W = tf.Variable(tf.zeros([imageSize, numberClasses]))
    
    # b means the bias. Create the bias, initialized by 0
    b = tf.Variable(tf.zeros([numberClasses]))
    
    # y means the output of calculation, softmax means the activation function is the multiple classes output
    # Formular is softmax((x * W) + b)
    labels = tf.nn.softmax(tf.matmul(x * W) + b)
    
    # Define the output prediction placeholder y_
    labels_ = tf.compat.v1.placeholder(tf.float32, [None, 10])
    
    # The parameter dictionary when creating the TensorFlow train model
    validationFeedDictionary = {images: imagesValidation, labels_: labelsValidation}
    testFeedDictionary = {images: imagesTest, labels_: labelsTest}
    
    # Define the loss function via activation the cross entropy of function softmax
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = labels_, logits = labels))
    
    # Define the Gradient Descent Optimizer, dradient descent according to the learning rate. In addition, the loss value becomes smaller and smaller when descending
    optimizer = tf.train.GradientDescendOptimizer(learningRate).minimize(cost)
    
    # Compare the correct prediction
    correctPrerdiction = tf.equal(tf.argmax(labels, 1), tf.argmax(labels_, 1))
    # Calculate the accuracy
    accuracy = tf.reduce_mean(tf.cast(correctPrerdiction, tf.float32))
    
    print("Accuracy = {}".format(accuracy))
    

imagesTrain, labelsTrain, imagesTest, labelsTest, imagesValidation, labelsValidation = Prepare()
CreateModel(imagesTest, labelsTest, imagesValidation, labelsValidation)
