import tensorflow as tf

def Read():
    
    mnist = tf.keras.datasets.mnist
    
    # import the dataset
    (imagesTrain, labelsTrain), (imagesTest, labelsTest) = mnist.load_data()
    
    return imagesTrain, labelsTrain, imagesTest, labelsTest

def Preprocess(imagesTrain, imagesTest):
    
    # Reshape
    imagesTrain4D = imagesTrain.reshape(imagesTrain.shape[0], 28, 28, 1).astype("float32")
    imagesTest4D = imagesTest.reshape(imagesTest.shape[0], 28, 28, 1).astype("float32")
    
    # Pixel standardization
    imagesTrain, imagesTest = imagesTrain4D / 225.0, imagesTest4D / 225.0
    
    return imagesTrain, imagesTest

def Prepare():
    
    imagesTrain, labelsTrain, imagesTest, labelsTest = Read()
    imagesTrain, imagesTest = Preprocess(imagesTrain, imagesTest)
    
    imagesValidation = imagesTrain[-10000:]
    labelsValidation = labelsTrain[-10000:]
    
    imagesTrain = imagesTrain[: -10000]
    labelsTrain = labelsTrain[: -10000]
    
    return imagesTrain, labelsTrain, imagesTest, labelsTest, imagesValidation, labelsValidation

 