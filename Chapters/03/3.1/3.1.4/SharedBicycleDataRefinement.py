from sklearn.model_selection import train_test_split

import sys
sys.path.insert(1, "../3.1.3/")

from SharedBicycleDataPreprocessing import Preprocess

def Refine():
	
	X_features, y_labels = Preprocess()
	X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size = 0.2, random_state = 42)
	
	test_size = (int)(X_test.shape[0] / 2)
	
	X_valid = X_test[: test_size]
	y_valid = y_test[: test_size]
	X_test = X_test[test_size:]
	y_test = y_test[test_size:]
	
	print("X_train.shape = {}, y_train.shape = {}".format(X_train.shape, y_train.shape))
	print("X_valid.shape = {}, y_valid.shape = {}".format(X_valid.shape, y_valid.shape))
	print("X_test.shape = {}, y_test.shape = {}".format(X_test.shape, y_test.shape))
	
Refine()