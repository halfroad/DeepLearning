import re
import numpy as np
from unicodedata import normalize
import string
import sys
import pickle

sys.path.insert(1, "../8.1.2/")
from French2EnglishInterpreterDataPreparation import Prepare

def RemovePhonetic(linePairs):
    
    # Mathch the string via Regular Expression
    printableRegularExpression = re.compile("[^{}]".format(re.escape(string.printable)))
    
    # Create a table to map the english punctuations, to remove the punctuations
    englishTable = str.maketrans("", "", string.punctuation)
    cleanedPairs = list()
    
    # Iterate the rows in France-English parallel sentences
    for pair in linePairs:
        
        cleanPair = list()
        
        for i, line in enumerate(pair):
            # Unicode and normalize the sentence
            line = normalize("NFD", line).encode("ascii", "ignore")
            line = line.decode("UTF-8")
            # Split the text by space
            line = line.split()
            # Lower
            line = [word.lower() for word in line]
            # Remove the punction in text
            line = [word.translate(englishTable) for word in line]
            # Remove the phonetic characters
            line = [printableRegularExpression.sub("", w) for w in line]
            # If it is not a alphabet, remove
            line = [word for word in line if word.isalpha()]
            
            cleanPair.append(" ".join(line))
            
        cleanedPairs.append(cleanPair)
        
    # Convert to numpy array
    cleanedPairs = np.array(cleanedPairs)
    
    print(cleanedPairs[: 15])
    
    with open("French2English.pkl", "wb") as f:
    
        pickle.dump(cleanedPairs, f)
    
def Split():
    
    with open("French2English.pkl", "rb") as f:
        
        dataset = pickle.load(f)
    
    length = 10000
    dataset = dataset[: length]
    
    # Randomly break the order
    np.random.shuffle(dataset)
    
    print(dataset[: 15])
    
    trainLength = length - 1500
    train, test = dataset[: trainLength], dataset[trainLength:]
    
    return dataset, train, test
    
def Save(sentences, fileName):
    
    with open(fileName, "wb") as f:
        
        pickle.dump(sentences, f)

linePairs = Prepare()
cleanedPairs = RemovePhonetic(linePairs)

print(string.printable)
print(string.punctuation)

dataset, train, test = Split()

Save(dataset, "French2EnglishTop10000.pkl")
Save(train, "French2EnglishTrain.pkl")
Save(test, "French2EnglishTest.pkl")

print("train.shape = {}, test.shape = {}".format(train.shape, test.shape))
