from sklearn.neural_network import MLPClassifier
import sklearn.metrics as metrics

import sys

sys.path.insert(1, "../2.1/")

from TrainDataRefinement import RefineTrainData

def PredictByMultipleLayerPerceptronClassifier():
	
	model = MLPClassifier(hidden_layer_sizes = 128, batch_size = 64, max_iter = 1000, solver = "adam")
	
	train_X, test_X, train_y, test_y = RefineTrainData()
	
	model.fit(train_X, train_y)
	
	train_predictions = model.predict(train_X)
	test_predictions = model.predict(test_X)
	
	print("Neutral Network Classifier Accuracy for train data is: {:.3f}".format(metrics.accuracy_score(train_y, train_predictions)))
	print("Neutral Network Classifier Accuracy for test data is: {:.3f}".format(metrics.accuracy_score(test_y, test_predictions)))
	
	y_score_nn = model.predict_proba(test_X)
	
	fpr_nn, tpr_nn, thresholds_nn = metrics.roc_curve(test_y, y_score_nn[:, 1])
	
	print("Neutral Network Classifier AUC is: {:.3f}".format(metrics.roc_auc_score(test_y, y_score_nn[:, 1])))
	
	return fpr_nn, tpr_nn

PredictByMultipleLayerPerceptronClassifier()