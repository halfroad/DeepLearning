from wordcloud import WordCloud
import matplotlib.pyplot as plt

import sys
sys.path.insert(1, "../4.1.2/")
from LotteryWinningPredictionPreparation import Prepare

def Draw(text):
    
    # Generate the object of Word Cloud
    # Parameter min_font_size: the minimum font size of the word
    # Parameter max_font_size: the maximum font size of the word
    # Parameter width: the width of the image drawn
    # Parameter width: the height of the image drawn
    wordCloud = WordCloud(min_font_size = 5, max_font_size = 200, width = 1200, height = 1000).generate(text)
    
    # Draw the figure
    plt.figure(figsize = (15, 8))
    plt.imshow(wordCloud, interpolation = "bilinear")
    
    # To hide axises of X and Y on the figure
    plt.axis("off")
    
    # Show the figure
    plt.show()

df, winningNumbers, notWinningNumbers = Prepare()

Draw(" ".join(["i" + str(n) for n in winningNumbers]))