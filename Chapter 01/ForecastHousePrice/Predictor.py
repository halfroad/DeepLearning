import pandas as pd

# Load the dataset
data = pd.read_csv('../MyBook/Chapter-1-Housing-Price-Prediction/housing.csv')

print("The housing data of Boston has {} rows and {} columns.".format(data.shape[0], data.shape[1]))

# First 5 rows
print(data.head())

data.describe()