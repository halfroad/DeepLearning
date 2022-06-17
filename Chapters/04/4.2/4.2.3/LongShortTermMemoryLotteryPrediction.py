import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from TimeSeriesLotteryInfrastructure import ClearSession

def Train(X, y, clearSessionNeeded = False):
    
    # Whether to clear the session
    
    if clearSessionNeeded:
        ClearSession()
        
    # Define the model
    model = Sequential()
    
    model.add(LSTM(50, activation = "relu", input_shape = (3, 1)))
    model.add(Dense(1))
    
    model.summary()
    
    model.compile(optimizer = "adam", loss = "mse")
    
    # Train the model
    model.fit(X, y, verbose = True)
    
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