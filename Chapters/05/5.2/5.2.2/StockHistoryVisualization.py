import numpy as np
import matplotlib.pyplot as plt
import os

import sys
sys.path.insert(1, "../../5.2/5.2.1")
from StockPredictionPreparation import AcquireStock

def PlotBasicStockHistory(df, startDate, endDate, name):
    
    # Define the Adj. Close as the statisctical field
    statisticalAdjustClose = "Adj. Close"
    
    # Minimum of Adj. Close
    statisticalMinimum = min(df[statisticalAdjustClose])
    # Date of minimum of Adj. Close
    statisticalMinimumDate = df[df[statisticalAdjustClose] == statisticalMinimum]["Date"]
    # Convert to Date for the maximum of Adj. Close
    statisticalMinimumDate = statisticalMinimumDate[statisticalMinimumDate.index[0]].date()
    
    # Maximum of Adj. Close
    statisticalMaximum = max(df[statisticalAdjustClose])
    # Date of minimum of Adj. Close
    statisticalMaximumDate = df[df[statisticalAdjustClose] == statisticalMaximum]["Date"]
    # Convert to Date for the maximum of Adj. Close
    statisticalMaximumDate = statisticalMaximumDate[statisticalMaximumDate.index[0]].date()
    
    # Mean of Adj. Close
    statisticalMinimum = np.mean(df[statisticalAdjustClose])
    
    print("The Minimum of {} is on {}, the price is ${} USD".format(statisticalAdjustClose, statisticalMinimumDate, statisticalMinimum))
    print("The Maximum of {} is on {}, the price is ${} USD".format(statisticalAdjustClose, statisticalMaximumDate, statisticalMaximum))
    
    print("The price for {} on {} is ${} USD.".format(statisticalAdjustClose, endDate.date(), df.loc[df.index[-1], statisticalAdjustClose]))
    
    # Set the style for plot
    plt.style.use("default")
    
    # Draw firgure
    # Parameter 1: Axis X is the date
    # Parameter 2: Axis Y is the price of Statistical Adjust Close
    # Parameter 3: The color for drawn line is Red
    # Parameter 4: Width of line is 3
    # Parameter 5: Label
    plt.plot(df["Date"], df[statisticalAdjustClose], color = "r", linewidth = 3, label = statisticalAdjustClose)
    
    # Set the label for Axis X
    plt.xlabel("Date")
    
    # Set the label for Axis Y
    plt.ylabel("US $")
    
    # Set the title
    plt.title("{} Stock History".format(name))
    
    # Enable the grid
    plt.grid()
    
    plt.show()
    

# Basic Trend
name = "BIDU"
baiduStock, minimumDate, maximumDate = AcquireStock(name)

PlotBasicStockHistory(baiduStock, minimumDate, maximumDate, name)

apiKey = os.path.expanduser("~") + "/.quandl_apikey"
    
os.remove(apiKey)