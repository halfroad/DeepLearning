from keras.utils import pad_sequences
from keras import utils
import numpy as np

import sys
sys.path.insert(1, "../../8.1/8.1.2/")
from French2EnglishInterpreterDataPreparation import Prepare

sys.path.insert(1, "../../8.1/8.1.3/")
from France2EnglishInterpreterPhoneticRemoval import RemovePhonetic, Split

sys.path.insert(1, "../8.2.1/")
from FrenchEnglishTokenization import CreateTokenizer, MaximumLength

def EncodeSequences(tokenizer, length, lines):
    
    X = tokenizer.texts_to_sequences(lines)
    
    # Pad sequences, the sentence will be padded by 0 if it is not long enough
    # maxlen means the max length of sequence
    # Padding can be by pre and post, means padding before and padding after
    X = pad_sequences(X, maxlen = length, padding = "post")
    
    return X

def EncodeOutput(sequences, vocabularySize):
    
    ylist = list()
    
    # Iterate the sequence
    for sequence in sequences:
        
        # Encoding by one-hot
        encoded = utils.to_categorical(sequence, num_classes = vocabularySize)
        
        ylist.append(encoded)
        
    # Convert to np array
    y = np.array(ylist)
    y = y.reshape(sequences.shape[0], sequences[1], vocabularySize)
    
    return y

def PrepareTrainTestData():
    
    linePairs = Prepare()
    cleanedPairs = RemovePhonetic(linePairs)

    dataset, train, test = Split()

    # Prepare the tokeonizer for French
    frenchTokenier = CreateTokenizer(dataset[:, 1])
    frenchVocabularySize = len(frenchTokenier.word_index) + 1
    frenchLength = MaximumLength(dataset[:, 1])

    # Prepare the tokeonizer for English
    englishTokenier = CreateTokenizer(dataset[:, 0])
    englishVocabularySize = len(englishTokenier.word_index) + 1
    englishLength = MaximumLength(dataset[:, 0])
    
    # Prepare data for train and test
    X_train = EncodeSequences(frenchTokenier, frenchLength, train[:, 1])
    y_train = EncodeSequences(englishTokenier, englishLength, train[:, 0])
    y_train = EncodeOutput(y_train, englishVocabularySize)
    
    X_test = EncodeSequences(frenchTokenier, frenchLength, test[:, 1])
    y_test = EncodeSequences(englishTokenier, englishLength, test[:, 0])
    y_test = EncodeOutput(y_test, englishVocabularySize)
    
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = PrepareTrainTestData()

print("X_train.shape = {}, y_train.shape = {}, X_test.shape = {}, y_test.shape = {}".format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))
