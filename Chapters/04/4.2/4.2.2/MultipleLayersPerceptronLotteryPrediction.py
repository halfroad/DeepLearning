from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

import sys
sys.path.insert(1, "../../4.1/4.1.2/")
from LotteryWinningPredictionPreparation import Prepare

sys.path.insert(1, "../../4.1/4.1.4/")
from SplitWinningNumbersSequences import SplitTrainRecords
from SplitWinningNumbersSequences import SplitSequence

def Predict(train_X, train_y, test_X, test_y):
    
    # Create the object of Multiple Layers Perceptron Classifier
    model = MLPClassifier(hidden_layer_sizes = 128, batch_size = 64, solver = "adam", verbose = True)
    
    # Train the model
    model.fit(train_X, train_y)
    
    # Predict the train data
    predictTrain = model.predict(train_X)
    # Predict the test data
    predictTest = model.predict(test_X)
    
    print("The accuracy of the Neural Network Classifier for train is {}.".format(accuracy_score(train_y, predictTrain)))
    print("The accuracy of the Neural Network Classifier for test is {}.".format(accuracy_score(test_y, predictTest)))


df, winningNumbers, notWinningNumbers = Prepare()
fearures = df["中奖号码"]
X, y = SplitSequence(fearures.values, 3)
train_X, test_X, train_y, test_y = SplitTrainRecords(X, y)

Predict(train_X, train_y, test_X , test_y)