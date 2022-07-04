import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.utils import pad_sequences

def Analyse():

    # To ensure the recurrence
    np.random.seed(7)
    
    topWords = 5000
    
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = topWords)
    
    # Set the maximum length is 500 for review
    reviewMaximumLength = 500
    
    # For those review that the length is less than 500, fill out by 0, and will be trancated if length overtakes 500
    X_train = pad_sequences(X_train, maxlen = reviewMaximumLength)
    X_test = pad_sequences(X_test, maxlen = reviewMaximumLength)
    
    print("X_train.shape = {}, X_test.shape = {}".format(X_train.shape, X_test.shape))
    
    return topWords, X_train, y_train, X_test, y_test, reviewMaximumLength
    

def CreateModel(topWords, reviewMaximumLength):
    
    embeddingVectorLength = 32
    model = Sequential()
    
    # Add input layer
    model.add(Embedding(topWords, embeddingVectorLength, input_length = reviewMaximumLength))
    # ADd LSTM hidden layer
    model.add(LSTM(100))
    # Add output layer (Full connected layer), 2 calssfication question, use sigmoid activation method
    model.add(Dense(1, activation = "sigmoid"))
    
    # Compile the model, 2 classification question, use binary cross entropy to compute loss
    model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])
    
    model.summary()
    
    return model


def Train():
    
    topWords, X_train, y_train, X_test, y_test, reviewMaximumLength = Analyse()
    
    model = CreateModel(topWords, reviewMaximumLength)
    
    model.fit(X_train, y_train, epochs = 3, batch_size = 64)
    
    scores = model.evaluate(X_test, y_test, verbose = 0)
    
    print("Accuracy: {}".format(scores[1] * 100))
    
    
Train()
    