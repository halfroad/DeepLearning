import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

def RefineTrainData():

	features = pd.read_csv("../../../MyBook/Chapter-2-Titanic-Survival-Exploration/titanic_dataset.csv")

	y_train = features["Survived"]
	X_train = features.drop("Survived", axis = 1)

	head = X_train.head()

	print(head)
	print("X_train.shape = {}, y_train.shape= {}".format(X_train.shape, y_train.shape))

	X_train.info()
	print(X_train.isnull().sum())

	sb.histplot(X_train["Age"].dropna(), kde = True)

	plt.title("Ages Distribution")
	plt.show()

	X_train["Age"].replace(np.nan, np.nanmedian(X_train["Age"]), inplace = True)

	sb.histplot(X_train["Age"], kde = True)

	plt.title("Ages Distribution")
	plt.show()

	X_train.drop("Cabin", axis = 1, inplace = True)

	sb.countplot(x = "Embarked", data = X_train)

	plt.title("Embarked Distribution")
	plt.show()

	X_train["Embarked"].replace(np.nan, "S", inplace = True)

	nanFares = X_train[np.isnan(X_train["Fare"])]

	print(nanFares)

	pclass3Fares = X_train.query("Pclass == 3 & Embarked == 'S'")["Fare"]

	pclass3Fares = pclass3Fares.replace(np.nan, 0)

	print(pclass3Fares)

	medianFare = np.median(pclass3Fares)

	X_train.loc[X_train["PassengerId"] == 1044, "Fare"] = medianFare

	X_train["Sex"].replace(["male", "female"], [1, 0], inplace = True)

	print(X_train.isnull().sum())

	X_train = pd.get_dummies(X_train)

	print("X_train.shape = {}, y_train.shape = {}".format(X_train.shape, y_train.shape))

	train_X, test_X, train_y, test_y = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42, shuffle = True)

	print("train_X.shape = {}, train_y.shape = {}".format(train_X.shape, train_y.shape))
	print("test_X.shape = {}, test_y.shape = {}".format(test_X.shape, test_y.shape))
	
	return train_X, test_X, train_y, test_y
