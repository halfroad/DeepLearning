import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

import sys
sys.path.insert(1, "../../5.2/5.2.1")
from StockPredictionPreparation import AcquireStock


def PlotPotentialDividend(df, startDate, endDate, name, shareQuantity = 1):
    
    # Acquire the first price
    firstOpenPrice = float(df[df["Date"] == startDate]["Adj. Open"])
    EndClosePrice = float(df[df["Date"] == endDate]["Adj. Close"])
    
    # Calculate the daily devidend = (Price of Adj. Clsoe - Open Price) * shareQuantity
    df["dividends"] = (df["Adj. Close"] - firstOpenPrice) * shareQuantity
    
    # Calculate the accumulative dividends
    totalDividends = (EndClosePrice - firstOpenPrice) * shareQuantity
    
    # Print the outputs
    print("From {} to {}, {} shares hold, total dividends is {}".format(startDate, endDate, shareQuantity, totalDividends))
    
    plt.style.use("default")
    
    plt.plot(df["Date"], df["dividends"], color = "m", linewidth = 3)
    
    plt.xlabel("Date")
    plt.ylabel("Dividends $")
    
    plt.title("Dividends for Shares from {} to {}".format(startDate, endDate))
    
    # Calculate the positions of digits
    horizontalTextLocation = (endDate - pd.DateOffset(months = 1)).date()
    verticalTextLocation = totalDividends + (totalDividends / 40)
    
    plt.text(horizontalTextLocation, verticalTextLocation, "$ {}".format(int(totalDividends)), color = "g", size = 15)
    
    plt.grid()
    plt.show()
    
# Basic Trend
name = "BIDU"
baiduStock, minimumDate, maximumDate = AcquireStock(name)

PlotPotentialDividend(baiduStock, minimumDate, maximumDate, name, 100)

startDate = np.datetime64("2012-08-07")
endDate = np.datetime64("2013-03-05")

# Acquire the stocks with the specific period
baiduStockLowerPricePhase = baiduStock[(baiduStock["Date"] >= startDate) & (baiduStock["Date"] <= endDate)]

PlotPotentialDividend(baiduStockLowerPricePhase, startDate, endDate, "BIDU", 100)

apiKey = os.path.expanduser("~") + "/.quandl_apikey"
    
os.remove(apiKey)
    
    