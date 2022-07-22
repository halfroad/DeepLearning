from keras import Sequential
from keras.layers import Dense, Embedding, LSTM, RepeatVector, TimeDistributed

import sys
sys.path.insert(1, "../8.2.2/")
from FrenchEnglishInterpreterEncodePad import PrepareTrainTestData

def Create(sourceVocabulary, targetVocabulary, sourceTimeSteps, targetTimeSteps, nUnits):
    
    model = Sequential()
    
    model.add(Embedding(sourceVocabulary, nUnits, input_length = sourceTimeSteps, mask_zero = True))
    model.add(LSTM(nUnits))
    model.add(RepeatVector(targetTimeSteps))
    model.add(LSTM(nUnits, return_sequences = True))
    
    # Add the output layer, the length of the layer is size of English vocabulary
    model.add(TimeDistributed(Dense(targetVocabulary, activation = "softmax")))
    
    return model

X_train, y_train, X_test, y_test, frenchVocabularySize, frenchLength, englishVocabularySize, englishLength = PrepareTrainTestData()

# Create the model
model = Create(frenchVocabularySize, englishVocabularySize, frenchLength, englishLength, 256)
# Compile the model
model.compile(optimizer = "adam", loss = "category_crossentropy")

# Preview the structure of model
model.summary()