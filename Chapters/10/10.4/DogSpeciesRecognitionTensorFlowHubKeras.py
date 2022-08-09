import tensorflow as tf
import keras
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def DownloadImageNetDogs():
    
    path = keras.utils.get_file("Images", "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar", untar = True)
    
    #os.rmdir(path)
    
    print(path)

    
def Prepare():
    
    inputImageSize = (224, 224)
    
    imageDataGenerator = ImageDataGenerator(rescale = 1 / 255)
    imageFlow = imageDataGenerator.flow_from_directory("Images", target_size = inputImageSize, batch_size = 32, color_mode = "rgb", class_mode = "categorical")
    
    for imageBatch, labelBatch in imageFlow:
        
        print("Image batch shape: ", imageBatch.shape)
        print("Label batch shape: ", labelBatch.shape)
        
        break
    
    stepsPerEpoch = imageFlow.samples // imageFlow.batch_size
    
    print("There are {} image samples, Batch Size is {}".format(imageFlow.samples, imageFlow.batch_size))
    print("A comprehensive train requires {} times.".format(stepsPerEpoch))
    
    return imageFlow, inputImageSize, stepsPerEpoch
    
def LoadPretrainModel(x):
    
    # featureExtractorURL = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5"
    featureExtractorURL = "imagenet_mobilenet_v2_100_224_feature_vector_5"
    featureExtractorModule = hub.Module(featureExtractorURL)
    
    return featureExtractorModule(x)

def CreateModel(imageFlow, inputImageSize):
    
    featuresExtractorLayer = keras.layers.Lambda(LoadPretrainModel, input_shape = inputImageSize + (3, ))
    featuresExtractorLayer.trainable = False
    
    model = tf.keras.Sequential([featuresExtractorLayer, tf.keras.layers.Dense(imageFlow.num_classes, activation="softmax")])
    
    print(model.summary())
    
    return model

# DownloadImageNetDogs()
imageFlow, inputImageSize, stepsPerEpoch = Prepare()
CreateModel(imageFlow, inputImageSize)
# LoadPretrainModel(1)
