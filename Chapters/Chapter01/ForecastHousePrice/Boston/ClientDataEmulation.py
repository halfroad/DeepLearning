from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from GridSearchModel import GridSearchFitModel

def PredictHousingPrice(X, y, fitter):
    # Iterate 10 times
    epochs = 10
    # Store predicted price
    y_predict_test_price = None
    
    # Split the train data and test data, 20% of the data is for test, remaining 80% as train data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    
    # Iterate train
    for epoch_i in range(epochs):
        
        # Return the optimised model according to the train model
        reg = fitter(X_train, y_train)
        
        # Predict the test data
        predicted_price = reg.predict(X_test)
        
        # Store the predicted results
        y_predict_test_price = predicted_price
        
        print("Iterate {} time(s)".format(epoch_i))
              
    return y_test, y_predict_test_price


data = pd.read_csv('../../../../MyBook/Chapter-1-Housing-Price-Prediction/housing.csv')

# acquire the housing prices
prices = data["MEDV"]

#acquire the features of house
features = data.drop('MEDV', axis=1)


y_true_price, y_predict_price = PredictHousingPrice(features, prices, GridSearchFitModel)

medv = pd.Series(y_true_price).reset_index().drop("index", axis = 1).head()

print(medv)

head = pd.Series(y_predict_price).head()

print(head)


        