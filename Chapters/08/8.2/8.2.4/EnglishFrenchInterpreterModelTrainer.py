from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as bk

import sys

sys.path.insert(1, "../8.2.2/")
from FrenchEnglishInterpreterEncodePad import PrepareTrainTestData

sys.path.insert(1, "../8.2.3/")
from EnglishFrenchInterpreterLongShortTermMemoryModel import Create

def Train(model, X_train, y_train, X_test, y_test, modelFileName):
    
    modelCheckPoint = ModelCheckpoint(modelFileName, monitor = "val_loss", verbose = 1, save_best_only = True, mode = "min")
    earlyStopCallback = EarlyStopping(monitor = "val_loss", patience = 3)
    
    # Train the model    
    model.fit(X_train, y_train, epochs = 50, batch_size = 64, validation_data = (X_test, y_test), callbacks = [modelCheckPoint, earlyStopCallback], verbose = 2)

modelFileName = "TranslatorWeightsModel.h5"
X_train, y_train, X_test, y_test, frenchVocabularySize, frenchLength, englishVocabularySize, englishLength = PrepareTrainTestData()

# Create the model
model = Create(frenchVocabularySize, englishVocabularySize, frenchLength, englishLength, 256)
# Compile the model
model.compile(optimizer = "adam", loss = "categorical_crossentropy")

bk.clear_session()

Train(model, X_train, y_train, X_test, y_test, modelFileName)