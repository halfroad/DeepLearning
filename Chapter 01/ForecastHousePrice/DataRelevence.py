import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

data = pd.read_csv('../MyBook/Chapter-1-Housing-Price-Prediction/housing.csv')

# acquire the housing prices
prices = data["MEDV"]

# regplot(): Draw the model of Linear Regression graph according to data

# Parameter 0: Axis x, indicates the average number of rooms for housing
# Parameter 1: Axis y, the prices of housing
# Parameter 2: the color to be drawn
sb.regplot(data["RM"], prices, color="red")

# Draw
plt.show()