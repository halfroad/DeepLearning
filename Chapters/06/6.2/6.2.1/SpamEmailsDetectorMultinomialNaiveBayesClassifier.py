from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

import sys
sys.path.insert(1, "../../6.1/6.1.2/")
from SpamEmailNaturalLanguageToolkit import ProcessByNaturalLanguageToolkit
from SpamEmailNaturalLanguageToolkit import SplitTrainTest


def Train():
    
    df = ProcessByNaturalLanguageToolkit()
    X_train, X_test, y_train, y_test = SplitTrainTest(df)
    
    # Create the object of vectorizer
    countVectorizer = CountVectorizer()
    
    # Convert the train and test set for the convenience of train
    X_TrainTextCounts = countVectorizer.fit_transform(X_train.values)
    
    # Convert the test set inyo vector of documentive item, for the convenience of cherry pick-up the annotation counts from orginal texts when testing
    X_TestTextCounts = countVectorizer.transform(X_test.values)
    
    # Test the result
    nbClassifier = MultinomialNB()
    # Train the model
    nbClassifier.fit(X_TrainTextCounts, y_train)
    
    # Predict
    predictions = nbClassifier.predict(X_TrainTextCounts)
    score = accuracy_score(y_test, predictions)
    
    print("The accuracy score is {:.7}%.".format(score * 100))
    
    goodEmail = ["Jin Hui, it is an important that our meeting will start at 3pm tommorow."]
    goodEmailCounts = countVectorizer.transform(goodEmail)
    prediction1 = nbClassifier.predict(goodEmailCounts)
    
    print("The result of the prediction is {}, it is a normal email".format(prediction1))
    
    spamEmail = ["Due to her husband passed away, she is looking for somebody to inherit the legacy of $100 million."]
    spamEmailCounts = countVectorizer.transform(spamEmail)
    prediction2 = nbClassifier.predict(spamEmailCounts)
    
    print("The result of the prediction is {}, it is a spam email".format(prediction2))


Train()