import matplotlib.pyplot as plt

# Set the default style for plot
plt.style.use("default")

import sys
sys.path.insert(1, "../4.1.2")
from LotteryWinningPredictionPreparation import Prepare

def QueryByDate(df, beginDate, endDate):
    
    """
    Define the function to query the winning lottery by dates
    """
    
    # Set the dates condition for the query
    mask = (df["开奖日期"] >= beginDate) & (df["开奖日期"] <= endDate)
    
    # Slice off via loc
    
    return df.loc[mask]

def DrawChart(X, y, title):
    
    """
    Draw the chart, X is the lottery revelation date, y is the winning number 
    """
    
    # Set the figure size
    plt.figure(figsize = (14, 5))
    
    # Draw the figure via X and y
    plt.plot(X, y)
    
    # Set the title
    plt.title(title)
    
    # Set the label for X axis
    plt.xlabel("Numbers of Perioids")
    
    # Set the label for Y axis
    plt.ylabel("Winning Number")
    
    # Show te grid
    plt.grid()
    
    plt.show()
    

# Begin date
startDate = "2018-01-01"
# End date
endDate = "2018-07-31"

df, winningNumbers, notWinningNumbers = Prepare()
dfOfYear2018 = QueryByDate(df, startDate, endDate)

print(dfOfYear2018.head())

print("From {} to {}, there are {} winning lottery".format(startDate, endDate, len(dfOfYear2018)))

title2018 = "Lottery 3D Winning Numbers, Date from {} to {}.".format(startDate, endDate)
DrawChart(dfOfYear2018["期号"], dfOfYear2018["中奖号码"], title2018)

# Begin date
startDate = "2017-01-01"
# End date
endDate = "2017-12-31"

dfOfYear2017 = QueryByDate(df, startDate, endDate)

print(dfOfYear2017.head())

print("From {} to {}, there are {} winning lottery".format(startDate, endDate, len(dfOfYear2017)))

title2017 = "Lottery 3D Winning Numbers, Date from {} to {}.".format(startDate, endDate)
DrawChart(dfOfYear2017["期号"], dfOfYear2017["中奖号码"], title2017)

