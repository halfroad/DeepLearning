import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os

import sys
sys.path.insert(1, "../../5.2/5.2.1")
from StockPredictionPreparation import AcquireStock

def Train(stockHistory, days = 0, weeklyReasonality = False, monthlyReasonality = False):

    # Create a model of prophet
    model = Prophet(daily_seasonality = False, weekly_seasonality = weeklyReasonality, yearly_seasonality = True, changepoint_prior_scale = 0.05)
    
    # Add monthly reasonality
    if monthlyReasonality:
        model.add_seasonality(name = "monthly", period = 30.5, fourier_order = 5)
        
    # Train the model
    model.fit(stockHistory)
    
    # Predict the stock in future according to the parameter days.
    future = model.make_future_dataframe(periods = days)
    
    # Predict
    future = model.predict(future)
    
    # Return the model and predicted values
    return model, future

def CreateModel(df, name, maximumDate, days = 0, weeklyReasonality = False, monthlyReasonality = False):
    
    # Find the records in recent 3 years
    stockHistory = df[df["Date"] > (maximumDate - pd.DateOffset(years = 3)).date()]
    
    # Train the model
    model, future = Train(stockHistory, days, weeklyReasonality, monthlyReasonality)
    
    # Set the default background for figure
    plt.style.use("default")
    
    # Initialize an object of plot
    figure, ax = plt.subplots(1, 1)
    
    # Set the size for the figure
    figure.set_size_inches(10, 5)
    
    # Draw the plot
    # Parameter 1: Date for the X axis
    # Parameter 2: Price for the Y axis
    # Parameter 3: Style of line,solid dot between lines
    # Parameter 4: width of line
    # Parameter 5: Alpha
    # Parameter 6: Marker size
    # Parameter 7: Label
    ax.plot(stockHistory["ds"], stockHistory["y"], "v-", linewidth = 1.0, alpha = 0.8, ms = 1.8, label = "Observations")
    
    # Draw the predicted values
    ax.plot(future["ds"], future["yhat"], "o-", linewidth = 1., label = "Modeled")
    
    # Draw an uncertain area
    ax.fill_between(future["ds"].dt.to_pydatetime(), future["yhat_upper"], future["yhat_lower"], alpha = 0.3, facecolor = "g", edgecolor = "k", linewidth = 1.0, label = "Confidence Interval")
    
    # Set the figure, loc == 2 means upper left, font size is 10
    plt.legend(loc = 2, prop = {"size": 10})
    
    # Set the title
    plt.title("{} Historical and Modeled Stock Price".format(name))
    
    # Set the x axis for Date
    plt.xlabel("Date")
    # Set the y axis for Price
    plt.ylabel("Price $")
    
    # Add the grid, width = 0.6, alpha = 0.6
    plt.grid(linewidth = 0.6, alpha = 0.6)
    
    # Show the plot
    plt.show()
    
    return model, future


    
# Basic Trend
name = "BIDU"
baiduStock, minimumDate, maximumDate = AcquireStock(name)

baiduStock["ds"] = baiduStock["Date"]
baiduStock["y"] = baiduStock["Adj. Close"]

model, future = CreateModel(baiduStock, name, maximumDate = maximumDate, monthlyReasonality = True)

apiKey = os.path.expanduser("~") + "/.quandl_apikey"
    
os.remove(apiKey)
