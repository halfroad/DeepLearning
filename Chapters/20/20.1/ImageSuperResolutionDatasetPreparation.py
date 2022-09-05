from glob import glob
from matplotlib import image

import matplotlib.pyplot as plt
import random


def Prepare():
    
    paths = glob("../../../Share/lfw/*/*")
    
    print(paths[: 10])
    
    return paths
    
def PlotRandomImages(paths, quantity):
    
    # Randomly select 25 image paths
    randomImages = random.sample(paths, 25)
    
    figure, axes = plt.subplots(nrows = 5, ncols = 5)
    
    figure.set_size_inches(8, 8)
    
    i = 0
    
    # Iterate rows 5
    for row in range(5):
        
        # Iterate columns 5
        for column in range(5):
            
            _image = image.imread(randomImages[i])
            ax = axes[row, column]
            
            # Show the image
            ax.imshow(_image)
            ax.grid(False)
            
            i += 1
            
    plt.show()
    
    
def Start():
    
    paths = Prepare()
    
    PlotRandomImages(paths, 25)
    
Start()