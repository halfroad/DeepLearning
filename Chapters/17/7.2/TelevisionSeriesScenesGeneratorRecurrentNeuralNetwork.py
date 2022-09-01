import numpy as np
import pickle
import tensorflow.compat.v1 as tf
import tf_slim as slim

slim.sequence_to_images

tf.disable_v2_behavior()

from collections import Counter

def CreateLookupTables(text):
    
    wordCounter = Counter(text)
    
    # Each id links to a word
    vocabularies = {counter: word for counter, word in enumerate(wordCounter)}
    
    # Each word links to an id
    identifiers = {word: counter for counter, word in vocabularies.items()}
    
    return vocabularies, identifiers

def CreatePunctuationsTokensLookup():
    
    '''

    Create a dictionary, map the punctuations to tokens
    
    '''
    
    tokensLookup = {".": "||Period||",
                  ",": "||Comma||",
                  "\"": "||Quatation_Mark||",
                  ";": "||Semicolon||",
                  "!": "||Exclaimation_Mark||",
                  "?": "||Question_Mark||",
                  "(": "||Left_Parentheses||",
                  ")": "||Right_Parentheses||",
                  "--": "||Dash||",
                  "\n": "||Return||"}
    
    return tokensLookup

def StorePreprocession(text, identifiers, vocabularies, tokens, storeURL):
    
    # storeURL = "../Exclusion/Datasets/PreprocessedEpisode.p"
    
    '''

    Store the preprocessed episode
    
    '''
    
    with open(storeURL, "wb") as f:
        
        pickle.dump((text, identifiers, vocabularies, tokens), f)
        
def LoadPreprocession(storeURL):
    
    '''

    Load the preprocessed episode
    
    '''
    
    return pickle.load(storeURL, mode = "rb")

def PreprocessScrips(text, storeURL = "../Exclusion/Datasets/PreprocessedEpisode.p"):
    
    '''

    Preprocess the scrips
    
    '''
    
    # Tokenize the punctuations
    
    tokens = CreatePunctuationsTokensLookup()
    
    for key, token in tokens.items():
        
        text = text.replace(key, " {} ".format(token))
        
    # Lower the text
    text = text.lower()
    
    # Split the text by default seperator: SPACE
    words = text.split()
    
    # Convert the text based episode data into lookup table
    vocabularies, identifiers = CreateLookupTables()
    
    # Acquire the id via word
    identifiedTexts = [identifiers[word] for word in words]
    
    StorePreprocessedEpisode(identifiedTexts, identifiers, vocabularies, tokens, storeURL)
    
    # Return the tuple
    return identifiedTexts, identifiers, vocabularies, tokens

def CreateInputs():
    
    _input = tf.placeholder(tf.int32, [None, None], name = "input")
    targets = tf.placeholder(tf.int32, [None, None], name = "targets")
    learningRate = tf.placeholder(tf.float32, name = "learning_rate")
    
    return (_input, targets, learningRate)

def InitializeRecurrentNeuralNetwork(batchSize, recurrentNeuralNetworkSize, numberOfLayers = 1):
    
    # Create BasicLSTMCell
    def CreateBasicLSTMCell(recurrentNeuralNetworkSize):
        
        cell = tf.nn.rnn_cell.LSTMCell(recurrentNeuralNetworkSize)
        
        return cell
    
    # Create multiple RNN Cells
    cell = tf.nn.rnn_cell.MultiRNNCell([CreateBasicLSTMCell(recurrentNeuralNetworkSize) for _ in range(numberOfLayers)])

    # Initialize the RNN Cell by zero
    initialState = cell.zero_state(batchSize, tf.float32)
    
    # Return a tensor which has the same shape with input
    initialState = tf.identity(input = initialState, name = "InitialState")
    
    return cell, initialState

def GetEmbeded(_input, vocabularySize, embededDimensions):
    
    # Create a tensor of random distribution
    embedding = tf.Variable(tf.random_uniform((vocabularySize, embededDimensions), -1, 1))
    
    # Find the id from an embeded tensor, return embeded matric
    embeded = tf.nn.embedding_lookup(embedding, _input)
    
    return embeded

def BuildRecurrentNeuralNetwork(cell, inputs):
    
    # Create a Recurrent Neural Network via Cell
    outputs, finalState = tf.nn.dynamic_rnn(cell, inputs, dtype = tf.float32)
    
    # Apply a final state
    finalState = tf.identity(input = finalState, name = "FinalState")
    
    return outputs, finalState

def BuildNeuralNetwork(cell, recurrentNeuralNetworkSize, _input, vocabularySize, embededDimensions):
    
    '''

    Build the Neural Network by above method
    
    Parameter 0 cell: Recurrent Neural Network cell
    Parameter 1 recurrentNeuralNetworkSize: Size of Recurrent Neural Network
    Parameter 2 _input: Input Data
    Parameter 3 vocabularySize: Size of Vocabulary
    Parameter 4 embededDimensions: Embeded Dimensions
    
    Return: Type of Tuple (Predictions, Finalstate)
    
    '''
    
    # Get the Embeded Layer
    embededLayer = GetEmbeded(_input, vocabularySize, embededDimensions)
    
    # Build RNN
    outputs, finalState = BuildRecurrentNeuralNetwork(cell, embededLayer)
    
    # Add a Fully-Connected layer, the size is vocabularySize
    predictions = tf.nn.layers.fully_connected(outputs, vocabularySize, activation_fn = None)
    
    # Return the predictions and final state
    return predictions, finalState

def CreateHyperParameters():
    
    # Epochs
    epochs = 200
    
    # Batch Size
    batchSize = 128
    
    # Size of Recurrent Neural Network
    recurrentNeuralNetworkSize = 256
    
    # Embeded Dimension
    embededDimensions = 500
    
    # Sequential Length
    sequentialLength = 10
    
    # Learning Rate
    learningRate = 0.01
    
    # Print statistics log every 50 times
    statisticsFrequency = 50
    
    
CreateInputs()
InitializeRecurrentNeuralNetwork(2, 2)
GetEmbeded(1, 1, 1)
BuildNeuralNetwork(1, 1, 1, 1, 1)
