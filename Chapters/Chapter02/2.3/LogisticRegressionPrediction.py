from sklearn.linear_model import LogisticRegression
from sklearn import metrics

import sys
sys.path.insert(1, "../2.1")

from TrainDataRefinement import RefineTrainData

def PredictByLogisticRegression():

	# Create the prediction model of Logistic Regression
	model = LogisticRegression(max_iter = 2000)
	
	train_X, test_X, train_y, test_y = RefineTrainData()
	
	# Train the model
	model.fit(train_X, train_y)
	
	print("Logistic Regression Accuracy for train data is: {:.3f}".format(model.score(train_X, train_y)))
	print("Logistic Regression Accuracy for test data is: {:.3f}".format(model.score(test_X, test_y)))
	
	y_score_lr = model.decision_function(test_X)
	
	print("Logistic Regression AUC is: {:.3f}".format(metrics.roc_auc_score(test_y, y_score_lr)))
	
	fpr_lr, tpr_lr, thresholds_lr = metrics.roc_curve(test_y, y_score_lr)
	
	return fpr_lr, tpr_lr
	
PredictByLogisticRegression()