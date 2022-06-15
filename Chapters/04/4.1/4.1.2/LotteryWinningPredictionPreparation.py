import pandas as pd
from collections import Counter

def Prepare():

    df = pd.read_csv("../../../../MyBook/Chapter-4-Lottery3D-Prediction/3d_lottery.csv")

    minimumWinningLotteryDate = min(df["开奖日期"])
    maximumWinningLotteryDate = max(df["开奖日期"])

    quantity = len(df)

    print("从{}到{}的福彩3D中奖记录，共有{}条。".format(minimumWinningLotteryDate, maximumWinningLotteryDate, quantity))

    winningNumbers = [[n if len(n) > 0 else '' for n in number.split(' ')] for number in df["中奖号码"]]
    winningNumbers = [int(''.join(digit)) for digit in winningNumbers]
    
    print(winningNumbers[: 10])
    
    df["中奖号码"] = winningNumbers
    
    print(df.head())
    
    counter = Counter(winningNumbers)
    
    # Show the top 15 frequent winning numbers
    top15 = counter.most_common(15)
    
    print(top15)
    
    mostCommon = counter.most_common()
    
    print("Revealed winning numbers combination: {}".format(len(mostCommon)))
    
    notWinningNumbers = []
    
    # Iterate the numbers within 1000
    for i in range(0, 1000):
        
        found = False
        
        # Compare this number against the winning numbers
        for number in mostCommon:
            if i == number[0]:
                found = True
                break
            
        if not found:
            notWinningNumbers.append(i)
            
        notWinningNumbers = PaddingZero(notWinningNumbers)
    
    print("These {} lottery numbers combination have not been the winning number, there are {}".format(len(notWinningNumbers), notWinningNumbers))
    
    return df, winningNumbers, notWinningNumbers

def PaddingZero(notWinningNumbers):
    
    results = []
    
    for number in notWinningNumbers:
        
        if len(str(number)) == 1:
               results.append("00" + str(number))
        elif len(str(number)) == 2:
                 results.append("0" + str(number))
        else:
                 results.append(str(number))
                 
    return results
                 
Prepare()