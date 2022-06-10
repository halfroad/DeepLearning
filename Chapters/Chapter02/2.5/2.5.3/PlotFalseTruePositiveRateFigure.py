import matplotlib.pyplot as plt

import sys
sys.path.insert(1, "../../2.1/")

from TrainDataRefinement import RefineTrainData

sys.path.insert(1, "../../2.2/")
from DecisionTreeClassfierPrediction import PredictByDecisionTreeClassifier

sys.path.insert(1, "../../2.3/")
from LogisticRegressionPrediction import PredictByLogisticRegression

sys.path.insert(1, "../../2.4/")
from GradientBoostingClassifierPrediction import PredictByGradientBoostingClassifier

sys.path.insert(1, "../../2.5/")
from  MultipleLayerPerceptronClassifierPrediction import PredictByMultipleLayerPerceptronClassifier

def PlotFalseTruePositiveRate():
	
	# Create a figure with the dimension 20 * 10
	figure = plt.figure(figsize = (20, 10))
	ax = figure.add_subplot(111)

	# Paraphrase the parameters for method plot()
	# Parameter 1: the data for Axis X
	# Parameter 2: the data for Axis Y
	# Parameter 3: the color of the line to be drawn
	# Parameter 4: width of the line
	# Parameter 5: title for the curve

	# Draw the curve for Decision Tree Classifier
	
	fpr_dt, tpr_dt = PredictByDecisionTreeClassifier()
	ax1 = ax.plot(fpr_dt, tpr_dt, c = 'c', lw = 2, label = "Decision Tree Classifier")
	
	# Draw the curve for Logistic Regression
	
	fpr_lr, tpr_lr = PredictByLogisticRegression()
	ax2 = ax.plot(fpr_lr, tpr_lr, c = 'y', lw = 2, label = "Logistic Regression")
	
	# Draw the curve for Gradient Boosting Classifier
	
	fpr_gb, tpr_gb = PredictByGradientBoostingClassifier()
	ax3 = ax.plot(fpr_gb, tpr_gb, c = 'r', lw = 2, label = "Gradient Boosting Classifier")
	
	# Draw the curve for Neutral Network
	
	fpr_nn, tpr_nn = PredictByMultipleLayerPerceptronClassifier()
	ax4 = ax.plot(fpr_nn, tpr_nn, c = 'b', lw = 2, label = "Neutral Network")
	
	ax.grid()
	
	lns = ax1 + ax2 + ax3 + ax4
	
	# Add the legend in the top-left corner
	ax.legend(handles = lns, loc = "upper left")
	
	# Show the figure
	plt.show()


PlotFalseTruePositiveRate()