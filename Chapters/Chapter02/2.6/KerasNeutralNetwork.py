import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras import utils as np_utils

import sys

sys.path.insert(1, "../2.1/")

from TrainDataRefinement import RefineTrainData

import pandas as pd
import numpy as np



def CreateKerasModel(X, y):
	
	# Create the model
	model = Sequential()
	
	# Initializes use the Truncated Normal
	initializers = keras.initializers.TruncatedNormal(mean = 0.0, stddev = 0.05, seed = None)
	
	# The dimension of input layer X.shape[1], 128 units
	model.add(Dense(input_dim = X.shape[1], units = 128, kernel_initializer = initializers, bias_initializer = "zeros"))
	
	# Add the Activation layer ReLU
	model.add(Activation("relu"))
	
	# Add layer Dropout
	model.add(Dropout(0.2))
	
	# Add the layer Full-Connected
	model.add(Dense(32))
	model.add(Activation("relu"))
	model.add(Dense(2))
	
	# The output is 1 or 0, so here the sigmoid activation method is utilized
	model.add(Activation("sigmoid"))
	
	# Using the binary corss entrophy and adam optimizer auto adjustment to compile
	model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])
	
	# One-hot encodes the train data y
	y_train_categorical = np_utils.to_categorical(y)
	
	# Train the model, epochs mean the times of train is 150, verbose means the log will be output when each batch is trained
	model.fit(X.values, y_train_categorical, epochs = 150, verbose = 1)
	
	return model

train_X, test_X, train_y, test_y = RefineTrainData()

keras_model = CreateKerasModel(train_X, train_y)

y_test_categorical = np_utils.to_categorical(test_y)

loss_and_accuracy = keras_model.evaluate(test_X.values, y_test_categorical)

print("Loss = {}, Accuracy = {}.".format(loss_and_accuracy[0], loss_and_accuracy[1]))

# Predict the suvival for the passengers versus the PassengerId
predictions_X = keras_model.predict(test_X.values)
predictions_classes = np.argmax(predictions_X, axis = 1)

# Align the results and Passenger IDs
submission = pd.DataFrame({
	"PassengerId": test_X["PassengerId"],
	"Survived": predictions_classes})

# show the top 15 records
print(submission[0: 15])