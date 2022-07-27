import matplotlib.pyplot as plt
import numpy as np

def Predict(model, image):
    
    # Predict the possibility of a givenimage, the result is the 10 possibilities of vector
    prediction = model.predict(image)
    
    # Preview the image
    coordinates = np.arange(prediction.shape[1])
    
    plt.bar(coordinates, prediction[0][: ])
    plt.xticks(coordinates, np.arange(10))
    
    plt.show()