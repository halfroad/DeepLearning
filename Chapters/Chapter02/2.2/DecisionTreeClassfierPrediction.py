from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(1, "../2.1")

from TrainDataRefinement import RefineTrainData

def PredictByDecisionTreeClassifier():
		
	# Create the decision tree model

	model = DecisionTreeClassifier()

	train_X, test_X, train_y, test_y = RefineTrainData()
	# Train model
	model.fit(train_X, train_y)

	# Predict the data of train
	train_predictions = model.predict(train_X)

	# Predict the data of test
	test_predictions = model.predict(test_X)

	# Calculate the accuracy of train
	train_accuracy = accuracy_score(train_y, train_predictions)

	# Calculate the accuracy of test
	test_accuracy = accuracy_score(test_y, test_predictions)

	print("The train accuracy is {}".format(train_accuracy))
	print("The test accuracy is {}".format(test_accuracy))

	y_score_dt = model.predict_proba(test_X)
	fpr_dt, tpr_dt, thresholds_dt = roc_curve(test_y, y_score_dt[:, 1])
	
	print("Decision Tree Classifier AUC is: {:.3f}".format(roc_auc_score(test_y, y_score_dt[:, 1])))

PredictByDecisionTreeClassifier()