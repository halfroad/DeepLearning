import keras
import os

def Prepare():
    
    path = keras.utils.get_file("Images", "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar", untar = True)
    
    os.rmdir(path)
    
    print(path)
    
    imageGenerator = keras.preprocessing.image.ImageDataGenerator(rescale = 1 / 255)
    
    
Prepare()