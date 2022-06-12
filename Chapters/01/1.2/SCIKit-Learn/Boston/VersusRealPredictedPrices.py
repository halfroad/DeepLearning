import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from GridSearchModel import GridSearchFitModel
from ClientDataEmulation import PredictHousingPrice

def PlotVersusPriceFigure(y_true_prices, y_predict_prices):
    
    # Create a window with dimension 10 * 7
    plt.figure(figsize = (10, 7))
    
    # Draw the true prices for figure 1
    X_show = np.rint(np.linspace(1, np.max(y_true_prices), len(y_true_prices))).astype(int)
    
    # Draw the line for figure 1, method plot():
    # Parameter 1: the values of X axis direction, the true prices from lowest to highest
    # Parameter 2: the values of Y axis direction, the true prices
    # Parameter 3: the style of the drawn line, i.e. "o-" means the circular dot, "-" means the solid line
    # Parameter 4: the color of the drawn line, here is cyan
    plt.plot(X_show, y_true_prices, "o-", color = "c")
    
    # Figure 2 is the predicted prices, be stacked over figure 1
    X_show_predicted = np.rint(np.linspace(1, np.max(y_predict_prices), len(y_predict_prices))).astype(int)
    
    # Draw the figure 2, method plot():
    # Parameter 1: the values of X axis direction, the predicted prices from lowest to highest
    # Parameter 2: the values of Y axis direction, the predicted prices
    # Parameter 3: the style of the drawn line, i.e. "o-" means the circular dot, "-"means the solid line
    # Parameter 4: The color of the drawn line, here is magenta
    plt.plot(X_show_predicted, y_predict_prices, "o-", color = "m")
    
    # Add title
    plt.title("Housing Prices Prediction")
    
    # Add legend
    plt.legend(loc="lower right", labels=["True Prices", "Predicted Prices"])
    
    # Add title for X axis
    plt.xlabel("House's Price Tendency By Array")
    
    # Add title for Y axis
    plt.ylabel("House's Price")
    
    # Show the plot
    plt.show()

# The versus for Boston house prices

data = pd.read_csv('../../../../MyBook/Chapter-1-Housing-Price-Prediction/housing.csv')

# acquire the housing prices
prices = data["MEDV"]

#acquire the features of house
features = data.drop('MEDV', axis=1)


y_true_prices, y_predict_prices = PredictHousingPrice(features, prices, GridSearchFitModel)


PlotVersusPriceFigure(y_true_prices, y_predict_prices)