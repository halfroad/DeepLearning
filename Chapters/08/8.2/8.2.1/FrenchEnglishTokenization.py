from keras.preprocessing.text import Tokenizer

import sys
sys.path.insert(1, "../../8.1/8.1.2/")
from French2EnglishInterpreterDataPreparation import Prepare

sys.path.insert(1, "../../8.1/8.1.3/")
from France2EnglishInterpreterPhoneticRemoval import RemovePhonetic, Split


def CreateTokenizer(lines):
    
    tokenizer = Tokenizer()
    
    tokenizer.fit_on_texts(lines)
    
    return tokenizer

def MaximumLength(lines):
    
    return max(len(line.split()) for line in lines)


linePairs = Prepare()
cleanedPairs = RemovePhonetic(linePairs)

dataset, train, test = Split()

# Prepare the tokeonizer for French
frenchTokenier = CreateTokenizer(dataset[:, 1])
frenchVocabularySize = len(frenchTokenier.word_index) + 1
frenchLength = MaximumLength(dataset[:, 1])
print("The maximum number of word in French is {}, number of words is {}".format(frenchLength, frenchVocabularySize))

# Prepare the tokeonizer for English
englishTokenier = CreateTokenizer(dataset[:, 0])
englishVocabularySize = len(englishTokenier.word_index) + 1
englishLength = MaximumLength(dataset[:, 0])

print("The maximum number of word in English is {}, number of words is {}".format(englishLength, englishVocabularySize))

print("Dataset.shape = {}".format(dataset.shape))
