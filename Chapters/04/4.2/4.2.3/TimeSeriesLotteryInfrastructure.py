from keras import backend as k
from keras.models import Sequential
from keras.layers import Dense

import numpy as np


def ClearSession():
    
    k.clear_session()
    

def Train(X, y, clearSessionNeeded = False):
    
    # Whether to clear the session of keras
    if clearSessionNeeded:
        ClearSession()
        
    # Define the model
    model = Sequential()
    
    # The length of input sequence is 3
    model.add(Dense(100, activation = "relu", input_dim = 3))
    
    # The length of output sequence is 1
    model.add(Dense(1))
    
    # Compile the model
    model.compile(optimizer = "adam", loss = "mse")
    model.summary()
    
    # Train the model
    model.fit(X, y, epochs = 2000, verbose = True)
    
    return model


X = np.array([[10, 20, 30], [20, 30, 40], [30, 40, 50], [40, 50, 60]])
y = np.array([40, 50, 60, 70])

model = Train(X, y, True)

xInput = np.array([50, 60, 70])
xInput = xInput.reshape((1, 3))

yHat = model.predict(xInput, verbose = True)

print("Predicted Value: {}.".format(yHat))