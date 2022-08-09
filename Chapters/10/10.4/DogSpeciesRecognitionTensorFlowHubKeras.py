import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def Download():
    
    path = keras.utils.get_file("Images", "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar", untar = True)
    
    #os.rmdir(path)
    
    print(path)

    
def Prepare():
    
    imageDataGenerator = ImageDataGenerator(rescale = 1 / 255)
    
    inputImageSize = (224, 224)
    imageFlow = imageDataGenerator.flow_from_directory("/root/.keras/datasets/Images", target_size = inputImageSize, batch_size = 32, color_mode = "rgb", class_mode = "categorical")
    
    for imageBatch, labelBatch in imageFlow:
        
        print("Image batch shape: ", imageBatch.shape)
        print("Label batch shape: ", labelBatch.shape)
        
        break
    
    stepsPerEpoch = imageFlow.samples // imageFlow.batch_size
    
    print("There are {} image samples, Batch Size is {}".format(imageFlow.samples, imageFlow.batch_size))
    print("A comprehensive train requires {} times.".format(stepsPerEpoch))

Download()
Prepare()