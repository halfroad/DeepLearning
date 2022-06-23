import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np

def VectorizeByNGram(X_train, y_train, X_test):
    
    # Key parameters to create TF-IDF vectorizer
    kwargs = {
        # n-gram range
        "ngram_range": (1, 2),
        "dtype": "int32",
        # ASCII or unicode, unicode means any characters can be tackled with, but handling the unicode is slower than ascii
        "strip_accents": "unicode",
        # Error handling when decoding characters, strick is the default
        "decode_error": "replace",
        # N-Gram could be word, also char. Here word is used to seperate
        "analyzer": "word",
        # the minimum frequence of the text
        "min_df": 2
        }
    
    vectorzier = TfidfVectorizer(**kwargs)
    
    # Vectorize the text inside the email in train data
    x_train = vectorzier.fit_transform(X_train)
    # Vectorize the text inside the email in test data
    x_test = vectorzier.transform(X_test)
    
    # Select the features against the maximum score of k
    # f_classif means the f value of square-difference provided by the calculation
    selector = SelectKBest(f_classif, k = x_train.shape[1])
    
    selector.fit(x_train, y_train)
    
    x_train = selector.transform(x_train)
    test_x = selector.transform(x_test)
    
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    
    return x_train.toarray(), x_test.toarray()