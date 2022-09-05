import math
import numpy as np

from PIL import Image

class MnistFaceGenerativeAdversarialNetworksDataset(object):
    
    def __init__(self, name, files):
        
        """

        Parameter 0 name: the category name of dataset
        Parameter 1 files: images array of dataset
        
        """
        
        LFW = "lfw"
        MNIST = "mnist"
        
        WIDTH = 28
        HEIGHT = 28
        
        # If the dataset is LFW dataset
        if name == LFW:
            
            self.mode = "RGB"
            
            channels = 3
            
        # If the dataset is MNIST dataset
        elif name == MNIST:
            
            self.mode = "L"
            
            channels = 1
            
        # All the paths of images
        self.files = files
        self.shape = len(files), WIDTH, HEIGHT, channels
        
    def GetBatches(self, batchSize):
        
        '''

        Batches when generating trains
        
        Parameter 0 batchSize: Size of batches
        
        Return: Batches
        
        '''
        
        MAXIMUM = 255
        
        i = 0
        
        while i + batchSize <= self.shape[0]:
            
            batch = self.GetBatch(self.files[i: i + batchSize], *self.shape[1: 3], mode = self.mode)
            
            i += batchSize
            
            yield batch / MAXIMUM - 0.5
            
    def GetBatch(self, files, width, height, mode):
        
        '''

        Parameter 0 files: Number of image files for this batch
        Parameter 1 width: Width of image
        Parameter 2 height: Height of image
        Parameter 3 mode: mode of image, L or RGB
        
        '''
        
        batch = np.array([self.AcquireImage(sample, width, height, mode) for sample in files]).astype(np.float32)
        
        # If the image is not 4 dimensions, convert to 4 dimensions
        if len(batch.shape) < 4:
            
            batch = batch.reshape(batch.shape + (1, ))
            
        return batch
    
    def AcquireImage(self, path, width, height, mode):
        
        '''

        Read the image from path
        
        Parameter 0 path: Path of image
        Parameter 1 width: Width of image
        Parameter 2 height: Height of image
        Parameter 3 mode: mode of image, L or RGB
        
        Return: Data of image
        
        '''
        
        image = Image.open(path)
        
        if image.size != (width, height):
            
            faceWidth = faceHeight = 108
            
            j = (image.size[0] - faceWidth) // 2
            i = (image.size[1] - faceHeight) // 2
            
            image = image.crop([j, i, j + faceWidth, i + faceHeight])
            image = image.resize([width, height], IMAGE.BILINEAR)
            
        return np.array(image.convert(mode))
        
        
        
        