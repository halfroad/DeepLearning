import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('../../MyBook/Chapter-1-Housing-Price-Prediction/housing.csv')

# acquire the housing prices
prices = data["MEDV"]

#acquire the features of house
features = data.drop('MEDV', axis=1)

x_train, x_test, y_train, y_test = train_test_split(features, prices, test_size=0.1, random_state=50)

print("x_train.shape = {}, y_train.shape = {}.".format(x_train.shape, y_train.shape))
print("x_test.shape = {}, y_test.shape = {}.".format(x_test.shape, y_test.shape))