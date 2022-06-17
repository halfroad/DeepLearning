import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

from TimeSeriesLotteryInfrastructure import ClearSession

def Train(X, y, clearSessionNeeded = False):
    
    # Whether to clear the session
    
    if clearSessionNeeded:
        ClearSession()
        
    # Define the model
    model = Sequential()
    
    # Add the 1D convolutional layer, depth is 64, kernel is 2, use relu to adjust the weight and inaccuracy, size of input is (3, 1)
    model.add(Conv1D(filters = 64, kernel_size = 2, activation = "relu", input_shape = (3, 1)))
    model.add(MaxPooling1D(pool_size = 2))
    model.add(Flatten())
    model.add(Dense(50, activation = "relu"))
    
    # Output layer is 1
    model.add(Dense(1))
    model.compile(optimizer = "adam", loss = "mse")
    
    model.summary()
    
    # Train the model
    model.fit(X, y, epochs = 1000, verbose = 0)
    
    return model
        

# Convert the train set from [samples, timesteps] size to [samples, timesteps, features] format
X = np.array([[10, 20, 30], [20, 30, 40], [30, 40, 50], [40, 50, 60]])
y = np.array([40, 50, 60, 70])

X = X.reshape((X.shape[0], X.shape[1], 1))

model = Train(X, y, True)

# Test purpose
xInput = np.array([50, 60, 70])
xInput = xInput.reshape((1, 3, 1))

yHat = model.predict(xInput, verbose = 0)

print("Predicted Value is {}".format(yHat))
              
              