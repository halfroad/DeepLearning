import sys
sys.path.insert(1, "../7.1.2/")

from MoviesCriticismSentimentPredictionPreparation import LoadReviews, Preprocess

sys.path.insert(1, "../7.1.3/")

from MoviesCriticismSentimentDatasetEncoder import FeatureVectorize, Filter

def Split(features, labels):
    
    # 80% for train
    splitTrainRatio = 0.8
    
    # Length of features vector
    featuresLenth = len(features)
    
    # Length of train dataset
    trainLength = int(featuresLenth * splitTrainRatio)
    
    train_X, validation_x = features[: trainLength], features[trainLength:]
    train_y, validation_y = labels[: trainLength], labels[trainLength:]
    
    # Halve the validation dataset
    validation_x_half = int(len(validation_x) / 2)
    
    # Seperate the validation dataset, a half for validation, another half for test
    validation_x, test_x = validation_x[: validation_x_half], validation_x[validation_x_half:]
    validation_y, text_y = validation_y[: validation_x_half], validation_y[validation_x_half:]
    
    # Print
    print("\t\t\tFeature Shapes:")
    print("Train set: \t\t{}".format(train_X.shape), "\nValidation set: \t{}".format(validation_x.shape), "\nTest set: \t\t{}".format(test_x.shape))
    


reviews, labels = LoadReviews()
reviews, string, words = Preprocess(reviews)
reviews_ints, labels = Filter(reviews, labels)
features = FeatureVectorize(reviews_ints)

Split(features, labels)