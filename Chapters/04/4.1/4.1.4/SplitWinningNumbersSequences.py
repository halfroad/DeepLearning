import numpy as np

import sys
sys.path.insert(1, "../4.1.2/")
from LotteryWinningPredictionPreparation import Prepare

def SplitSequence(sequence, steps):
    
    """
    Split the univaribale sequence
    """
    
    # Reverse the data, because the prediction will be commenced from 2004 with the upwards direction
    sequence = sequence[::-1]
    
    X, y = list(), list()
    
    for i in range(len(sequence)):
        
        # Find the designated 'length steps' mode, the length equals X plus y
        end = i + steps
        
        # If the last length of last mode overtakes the total length, the last mode will be ignored.
        if end > len(sequence) - 1:
            break
        
        # Grab the X and y for this time
        # Way to grab X: i means the start value, end means the length of mode of steps
        # Way to grab y: is the end
        sequenceX , sequenceY = sequence[i: end], sequence[end]
        
        # Append X and y to the array
        X.append(sequenceX)
        y.append(sequenceY)
        
    return np.array(X), np.array(y)

def PreviewSequence(X, y, top = 10):
    
    """
    Preview the features and objective sequence within top n
    """
    
    _X = X[len(X) - top:]
    _y = y[len(X) - top:]
    
    for i, v in enumerate(_X):
        print(v, _y[i])
        

def SplitTrainRecords(X, y):
    
    # The proportionate of test is 15%
    testRatio = 0.15
    
    # The entire length of dataset
    length = len(X)
    
    # The length of test set
    testLength = int(length * testRatio)
    
    # Use the trailing segment for the train data
    X_train, y_train = X[testLength:], y[testLength:]
    
    # Use the heading segment for the test data
    X_test, y_test = X[: testLength], y[:testLength]
    
    # Print
    print("X_train.shape = {}, y_train_shape = {}".format(X_train.shape, y_train.shape))
    print("X_test.shape = {}, y_test_shape = {}".format(X_test.shape, y_test.shape))
    
    return X_train, X_test, y_train, y_test
    
    
df, winningNumbers, notWinningNumbers = Prepare()
fearures = df["中奖号码"]

X, y = SplitSequence(fearures.values, 3)
PreviewSequence(X, y)
SplitTrainRecords(X, y)

