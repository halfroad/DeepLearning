import pickle
import numpy as np
from nltk.translate.bleu_score import corpus_bleu

import sys
sys.path.insert(1, "../8.2.1/")
from FrenchEnglishTokenization import CreateTokenizer, MaximumLength
sys.path.insert(1, "../8.2.2/")
from FrenchEnglishInterpreterEncodePad import EncodeSequences

from keras import models


def loadCleanSentences(fileName):
    
    return pickle.load(open(fileName, "rb"))

def Verify():
    
    # Load the dataset
    dataset = loadCleanSentences("French2EnglishTop10000.pkl")
    # Load the train data
    trainDataset = loadCleanSentences("French2EnglishTrain.pkl")
    # Load the test data
    testDataset = loadCleanSentences("French2EnglishTest.pkl")

    # Prepare the tokenizer for English
    englishTokenizer = CreateTokenizer(dataset[:, 0])
    # Prepare the tokenizer for French
    frenchTokenizer = CreateTokenizer(dataset[:, 1])

    # Maximum number of French sequence
    frenchLenth = MaximumLength(dataset[:, 1])

    # Encode the train set
    X_train = EncodeSequences(frenchTokenizer, frenchLenth, trainDataset[:, 1])
    # Encode the test set
    X_test = EncodeSequences(frenchTokenizer, frenchLenth, testDataset[:, 1])
    
    # Load the saved model
    model = models.load_model("../8.2.4/TranslatorWeightsModel.h5")
    
    return model, X_train, X_test, trainDataset, testDataset, englishTokenizer, frenchTokenizer
    
def WordForId(integer, tokenizer):
    
    # Find the word from dictionary of tokenizer for the specific index, and return the word
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
        
    return None

def PredictSequence(model, tokenizer, source):
    
    # Predict the Egnlish word according to the word in French
    prediction = model.predict(source, verbose = 0)[0]
    
    # Select the index of the most possible value according to the result of prediction in the array.
    # Finally, return all the values of most possible
    integers = [np.argmax(vector) for vector in prediction]
    target = list()
    
    # Find the word relies on the value for the most possible.
    for i in integers:
        
        word = WordForId(i, tokenizer)
        
        if word is None:
            
            break
        
        target.append(word)
        
    return " ".join(target)
    
def TestModel(model, tokenier, sources, rawDataset):
    
    actual, predicted = list(), list()
    
    # Iterate the sources
    for i, source in enumberate(sources):
        
        # Predict the decoded French
        source = source.reshape(1, source.shape[0])
        translation = PredictSequence(model, tokenier, source)
        
        rawTarget, rawSource = rawDataset[i], rawDataset[i]
        
        if i < 10:
            
            print("Source sentence = [{}], target sentence = [{}], predicted sentence = [{}]".format(rawSource, rawTarget, translation))

        actual.append(rawTarget.split())
        predicted.append(translation.split())
        
        # Calculate the score of BLEU
        print("BLEU-1: {}".format(corpus_bleu(actual, predicted, weights = (1.0, 0, 0, 0))))
        print("BLEU-2: {}".format(corpus_bleu(actual, predicted, weights = (0.5, 0.6, 0, 0))))
        print("BLEU-3: {}".format(corpus_bleu(actual, predicted, weights = (0.3, 0.3, 0.3, 0))))
        print("BLEU-4: {}".format(corpus_bleu(actual, predicted, weights = (0.25, 0.25, 0.25, 0.25))))

# Test the sequence of train data

model, X_train, X_test, trainDataset, testDataset, englishTokenizer, frenchTokenizer = Verify()

print("Train data:")

TestModel(model, englishTokenizer, X_train, trainDataset)


print("Test data:")

TestModel(model, englishTokenizer, X_test, testDataset)
        