import numpy as np
import pandas as pd
import math
from keras.models import load_model
import sys

sys.path.insert(1, "../../4.1/4.1.2/")
from LotteryWinningPredictionPreparation import Prepare

sys.path.insert(1, "../../4.1/4.1.4/")
from SplitWinningNumbersSequences import SplitSequence

sys.path.insert(1, "../4.2.3/")
from LongShortTermMemoryLotteryPrediction import Train as LSTMTrain



def Predict():
    
    df = pd.read_csv("../../../../MyBook/Chapter-4-Lottery3D-Prediction/3d_lottery.csv")
    winningNumbers = [[n if len(n) > 0 else '' for n in number.split(' ')] for number in df["中奖号码"]] 

    # Convert each number into integer
    winningNumbers = [[int(n) for n in t] for t in winningNumbers]
    
    # Grab the first column
    firstColumn = np.array(winningNumbers)[:, 0]
    # Grab the second column
    secondColumn = np.array(winningNumbers)[:, 1]
    # Grab the third column
    thirdColumn = np.array(winningNumbers)[:, 2]
    
    # Split and Prepare
    # Split the X, y from first column
    first_X, first_y = SplitFeatures(firstColumn)

    # Split the X, y from second column
    second_X, second_y = SplitFeatures(secondColumn)
    
    # Split the X, y from third column
    third_X, third_y = SplitFeatures(thirdColumn)
    
    # Define the file name of the stored train models
    firstModelPath = "first_model.h5"
    secondModelPath = "second_model.h5"
    thirdModelPath = "third_model.h5"
    
    # Train the model
    Train(first_X, first_y, firstModelPath)
    Train(second_X, second_y, secondModelPath)
    Train(third_X, third_y, thirdModelPath)
    
    # Firstly, load the model from current directory, then divide 10 winning records for test purpose, compare the actual records and predicted records eventually
    
    # Load the train model for first column
    firstDataModel = load_model(firstModelPath)
    # Load the train model for second column
    secondDataModel = load_model(secondModelPath)
    # Load the train model for third column
    thirdDataModel = load_model(thirdModelPath)
    
    # Divide the test records, choose 10 winning lotteries from trained records
    first_test_X, first_test_y = SplitFeatures(firstColumn[: 10])
    second_test_X, second_test_y = SplitFeatures(secondColumn[: 10])
    third_test_X, third_test_y = SplitFeatures(thirdColumn[: 10])
    
    # Use the LSTM network to train the model, before passing the data, the data needs to be reshaped from [samples, timesteps] to [samples, timesteps, features]
    first_test_X = first_test_X.reshape(first_test_X.shape[0], first_test_X.shape[1], 1)
    second_test_X = second_test_X.reshape(second_test_X.shape[0], second_test_X.shape[1], 1)
    third_test_X = third_test_X.reshape(third_test_X.shape[0], third_test_X.shape[1], 1)
    
    # Predict
    firstPredictedValue = firstDataModel.predict(first_test_X, verbose = True)
    secondPredictedValue = secondDataModel.predict(second_test_X, verbose = True)
    thirdPredictedValue = thirdDataModel.predict(third_test_X, verbose = True)
    
    # Compare the actual values and predicted values, the difference is huge
    # Heap the data of 3 columns horizontally
    final_predicted_X = np.vstack([firstPredictedValue[:, 0], secondPredictedValue [:, 0], thirdPredictedValue[:, 0]])
    final_target_X = np.vstack([first_test_y[: 10].tolist(), second_test_y[: 10].tolist(), third_test_y[: 10].tolist()])
    
    # Transpose the array for print conveniently
    final_predicted_X = final_predicted_X.transpose()
    final_target_X = final_target_X.transpose()
    
    for i, v in enumerate(final_predicted_X):
        
        print("Predicted: {} VS. Target: {}.".format([math.ceil(p) for p in final_predicted_X[i]], final_target_X[i])) 
    
    
def SplitFeatures(X, steps = 3):
    
    X, y = SplitSequence(X, steps)
    
    return X, y

def Train(X, y, modelPath):
    
    X = X.reshape(X.shape[0], X.shape[1], 1)
    # Use the LSTM to train
    lstmPredictorModel = LSTMTrain(X, y, True)
    lstmPredictorModel.save(modelPath)
    
    
Predict()