import quandl
import os
import pandas as pd

def Init():

    quandl.save_key("6Dwi87w5PR259FQACyD_")
    
def AcquireStock(name):
    
    # Acquire baidu stock data
    stock = quandl.get("WIKI/{}".format(name))
    
    # Set the Date as the first column
    stock = stock.reset_index(level = 0)
    
    apiKey = os.path.expanduser("~") + "/.quandl_apikey"
    
    os.remove(apiKey)
    
    return stock
    
Init()

name = "BIDU"
baiduStock = AcquireStock(name)

print(baiduStock.head())
print(len(baiduStock))

minimumDate = min(baiduStock["Date"])
maximumDate = max(baiduStock["Date"])

print("Baidu Stock is active from {} to {}.".format(minimumDate, maximumDate))

print(type(baiduStock))

baiduStock.to_csv("baiduStock.csv", index = False)

df = pd.read_csv("baiduStock.csv")

print(df.head())

