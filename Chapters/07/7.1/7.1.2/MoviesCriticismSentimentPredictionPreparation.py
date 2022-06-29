import numpy as np
from os import path
from string import punctuation


def LoadReviews():
    
    # Load the criticisms
    url = path.relpath("../../../MyBook/Chapter-7-Sentiment-Analysis/reviews.txt", "r")
        
    with open(url) as f:
        
        reviews = f.read()
    
    url = path.relpath("../../../MyBook/Chapter-7-Sentiment-Analysis/labels.txt", "r")
    
    with open(url, "r") as f:
        
        labels = f.read()
        
    return reviews, labels


def Preprocess(reviewsString):
    
    string = "".join([review for review in reviewsString if review not in punctuation])
    reviews = string.split('\n')
    
    string = " ".join(reviews)
    words = string.split()
    
    return reviews, string, words

reviews, labels = LoadReviews()

reviews, string, words = Preprocess(reviews)

print("Reviews = {}".format(reviews))
print("String = {}".format(string))
print("Words = {}".format(words))