import math
import sys
sys.path.insert(1, "../../4.1/4.1.2/")
from LotteryWinningPredictionPreparation import Prepare

sys.path.insert(1, "../../4.1/4.1.4/")
from SplitWinningNumbersSequences import SplitTrainRecords
from SplitWinningNumbersSequences import SplitSequence

sys.path.insert(1, "../4.2.3/")
from TimeSeriesLotteryInfrastructure import Train as MLPTrain
from LongShortTermMemoryLotteryPrediction import Train as LSTMTrain


df, winningNumbers, notWinningNumbers = Prepare()
fearures = df["中奖号码"]
X, y = SplitSequence(fearures.values, 3)
X_train, X_test, y_train, y_test = SplitTrainRecords(X, y)

# Train the model
mlpPredictorModel = MLPTrain(X_train, y_train, True)

# Predict
yHat = mlpPredictorModel.predict(X_test, verbose = True)

# Print top 10
for i in range(10):
    
    print("Actual Winning Number is {}, Predicted Number is {}".format(y_test[i], math.ceil(yHat[i])))
    

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

# Train the model
lstmPredictorModel = LSTMTrain(X_train, y_train, True)

# Predict
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
yHat = lstmPredictorModel.predict(X_test, verbose = True)

# Print top 10
for i in range(10):
    
    print("Actual Winning Number is {}, Predicted Number is {}".format(y_test[i], math.ceil(yHat[i])))
