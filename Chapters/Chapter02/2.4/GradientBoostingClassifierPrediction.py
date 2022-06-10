from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import sys

sys.path.insert(1, "../2.1/")
from TrainDataRefinement import RefineTrainData

def PredictByGradientBoostingClassifier():

	model = GradientBoostingClassifier(n_estimators = 500)

	train_X, test_X, train_y, test_y = RefineTrainData()

	model.fit(train_X, train_y)

	# Predict the train data
	train_predictions = model.predict(train_X)

	# Predict the test date
	test_predictions = model.predict(test_X)

	# Calculate the train accuracy
	train_accuracy = accuracy_score(train_y, train_predictions)

	# Calculate the test accuracy
	test_accuracy = accuracy_score(test_y, test_predictions)

	print("Gradient Boosting Accuracy for train data is: {:.3f}".format(train_accuracy))
	print("Gradient Boosting Accuracy for test data is: {:.3f}".format(test_accuracy))

	# Calculate the ROC curve
	y_score_gb = model.predict_proba(test_X)

	fpr_gb, tpr_gb, thresholds_gb = roc_curve(test_y, y_score_gb[:, 1])

	print("Gradient Boosting Classifier AUC is: {:.3f}".format(roc_auc_score(test_y, y_score_gb[:, 1])))
	
	return fpr_gb, tpr_gb

PredictByGradientBoostingClassifier()