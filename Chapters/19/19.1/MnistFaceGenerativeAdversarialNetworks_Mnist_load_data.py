from keras.datasets import mnist

def Prepare():
    
    (imagesTrain, labelsTrain), (imagesTest, labelsTest) = mnist.load_data()
    
    print("imagesTrain.shape = {}, labelsTrain.shape = {}, imagesTest.shape = {}, labelsTest.shape = {}".format(imagesTrain.shape, labelsTrain.shape, imagesTest.shape, labelsTest.shape))
    
    print(imagesTrain[: 10])
    
Prepare()