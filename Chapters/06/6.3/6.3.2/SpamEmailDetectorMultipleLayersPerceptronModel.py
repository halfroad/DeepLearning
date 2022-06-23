import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

import sys

sys.path.insert(1, "../../6.1/6.1.2/")
from SpamEmailNaturalLanguageToolkit import ProcessByNaturalLanguageToolkit, SplitTrainTest

sys.path.insert(1, "../6.3.1/")
from SpamEmailsDetectorVectorization import VectorizeByNGram

def Create(layers, units, dropoutRate, inputShape, numberOfClasses):
    
    # Create the model of Sequential
    model = Sequential()
    
    # Add input layer
    model.add(Dropout(rate = dropoutRate, input_shape = inputShape))
    
    # Add the hidden layer
    for _ in range(layers - 1):
        
        model.add(Dense(units = units, activation = "relu"))
        model.add(Dropout(rate = dropoutRate))
        
    # Add the output layer, because the final predicted values are 1 or 0, so the sigmoid function can be used for activation
    outputUnits = numberOfClasses - 1
    model.add(Dense(units = outputUnits, activation = "sigmoid"))
    
    return model

def Train(X_train, y_train, X_test, y_test):
    
    # Rate of leanring, here the scientic notation, is 0.001
    learningRate = 1e-3
    
    # Number of iterations
    epochs = 1000
    
    # Size of each batch
    batchSize = 128
    
    # Number of hidden layers
    layers = 2
    
    # Dimensions of output space
    units = 64
    
    # Rate of dropout for each layer
    dropoutRate = 0.2
    
    # Only 2 consequences, 0 or 1
    numberOfClasses = 2
    
    # Vectorize the content of email
    x_train, x_test = VectorizeByNGram(X_train, y_train, X_test)
    
    # Create the model of neural network
    model = Create(layers = layers, units = units, dropoutRate = dropoutRate, inputShape = x_train.shape[1: ], numberOfClasses = numberOfClasses)
    
    # Compile the model
    optimizer = tf.keras.optimizers.Adam(lr = learningRate)
    model.compile(optimizer = optimizer, loss = "binary_crossentropy", metrics = ["accuracy"])
    
    model.summary()
    
    # Create the callback of early stop when the model verifies the model, as is given that within twice, if the loss is not reduced, the train will stop
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 2)]
    
    # Train and verify the model
    history = model.fit(x_train, y_train, epochs = epochs, callbacks = callbacks, validation_data = (x_test, y_test), verbose = 2, batch_size = batchSize)
    
    _history = history.history
    
    accuracyValue = _history["val_acc"]
    _accuracyValue = accuracyValue[np.argmax(accuracyValue)]
    
    lossValue = _history["val_loss"]
    _lossValue = lossValue[np.argmax(lossValue)]
    
    print("Verified accuracy is {}, verfied loss is {}.".format(_accuracyValue, _lossValue))
          
    # Save the model
    model.save("spam_email_classifier_model.h5")
    
    return model, _history


    
df = ProcessByNaturalLanguageToolkit()

X_train, X_test, y_train, y_test = SplitTrainTest(df)

model, history = Train(X_train, y_train, X_test, y_test)

print(history.key())