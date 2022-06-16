from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import sys
sys.path.insert(1, "../../4.1/4.1.2/")
from LotteryWinningPredictionPreparation import Prepare

sys.path.insert(1, "../../4.1/4.1.4/")
from SplitWinningNumbersSequences import SplitTrainRecords
from SplitWinningNumbersSequences import SplitSequence

def Predict(train_X, train_y, test_X, test_y):
    
    # Create the object of Dicision Tree Classifier
    model = DecisionTreeClassifier()
    
    # Train the model
    model.fit(train_X, train_y)
    
    # Predict the train data
    predictTrain = model.predict(train_X)
    
    # Predict the train data
    predictTest = model.predict(test_X)
    
    # Calculate the accuracy for train set
    trainAccuracy = accuracy_score(train_y, predictTrain)
    
    # Calculate the accuracy for test set
    testAccuracy = accuracy_score(test_y, predictTest)
    
    # Print
    print("Train Accuracy is {}.".format(trainAccuracy))
    print("Test Accuracy is {}.".format(testAccuracy))
    

df, winningNumbers, notWinningNumbers = Prepare()
fearures = df["中奖号码"]
X, y = SplitSequence(fearures.values, 3)
train_X, test_X, train_y, test_y = SplitTrainRecords(X, y)

Predict(train_X, train_y, test_X , test_y) 
    