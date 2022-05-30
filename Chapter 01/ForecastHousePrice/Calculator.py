import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_csv('../MyBook/Chapter-1-Housing-Price-Prediction/housing.csv')

# the lowest price
minimum_price = np.min(data["MEDV"])

# the highest price
maximum_price = np.max(data["MEDV"])

# the means
mean_price = np.mean(data["MEDV"])

# the median price
median_price = np.median(data["MEDV"])

# the standard deviation
std_price = np.std(data["MEDV"])

# Print all the prices

print("The price statistics for Boston:")

print("The lowest price: ${}".format(minimum_price))
print("The highest price: ${}".format(maximum_price))
print("The mean price: ${}".format(mean_price))
print("The median price: ${}".format(median_price))
print("The standard deviation price: ${}".format(std_price))

# acquire the housing prices
prices = data["MEDV"]

#acquire the features of house
features = data.drop('MEDV', axis=1)