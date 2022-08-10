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

def CreateModel(imageFlow, inputImageSize):
    
    featureExtractorURL = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5"

    layers = [
        hub.KerasLayer(handle = featureExtractorURL, input_shape = inputImageSize + (3, ), trainable = False, name = "mobilenet_embedding"),
        tf.keras.layers.Dense(imageFlow.num_classes, activation="softmax")

        ]
    model = tf.keras.Sequential(layers, name = "DogSpeciesFeatures")
        
    print(model.summary())
    
    return model


class CollectionBatchStats(tf.keras.callbacks.Callback):
        
        def __init__(self):
            
            self.batch_losses = []
            self.batch_accuracies = []
            
            def onBatchEnd(self, batch, logs = None):
                
                self.batch_losses.append(logs["loss"])
                self.batch_accuracies.append(logs["accuracy"])

def TrainModel(model, imageFlow, stepsPerEpoch):
    
    model.compile(optimizer = keras.optimizers.Adam(), loss = "categorical_crossentropy", metrics = ["accuracy"])
    
    batchStats = CollectionBatchStats()
    
    model.fit((item for item in imageFlow), epochs = 5, steps_per_epoch = stepsPerEpoch, callbacks = [batchStats])
    

# DownloadImageNetDogs()
imageFlow, inputImageSize, stepsPerEpoch = Prepare()
model = CreateModel(imageFlow, inputImageSize)

TrainModel(model, imageFlow, stepsPerEpoch)
# LoadPretrainModel(1)
