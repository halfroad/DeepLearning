import pandas as pd
import sys
sys.path.append("../Boston")
from ClientDataEmulation import PredictHousingPrice
from GridSearchModel import GridSearchFitModel
from VersusRealPredictedPrices import PlotVersusPriceFigure

df = pd.read_csv("../../../../MyBook/Chapter-1-Housing-Price-Prediction/bj_housing.csv");
description = df.describe()

print(description)

prices = df["Value"]
features = df.drop("Value", axis = 1)

print(features.head())

y_true_prices, y_predict_prices = PredictHousingPrice(features, prices, GridSearchFitModel)

head = y_true_prices.reset_index().drop("index", axis = 1).head()

print(head)

head = pd.Series(y_predict_prices).head()

print(head)

PlotVersusPriceFigure(y_true_prices, y_predict_prices)