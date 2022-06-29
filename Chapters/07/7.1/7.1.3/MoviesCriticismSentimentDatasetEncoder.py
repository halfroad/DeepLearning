from collections import Counter
import numpy as np
import sys
sys.path.insert(1, "../7.1.2/")

from MoviesCriticismSentimentPredictionPreparation import LoadReviews, Preprocess

def showItems(dictionary, top = 10):
    
    index = 0
    
    for k, v in dictionary.items():
        
        if index >= top:
            break
        
        print("{}: {}". format(k, v))
        
        index += 1
        
def Filter(reviews, labels):

    reviews_ints = []
    
    # Calculate the reoccurances of the word
    wordCounter = Counter(words)
    # Sort the wordCounter by descending order
    sortedVocabulary = sorted(wordCounter, key = wordCounter.get, reverse = True)

    showItems(wordCounter)

    mostCommon = wordCounter.most_common(15)

    print(mostCommon)

    print(sortedVocabulary[: 15])

    vocabulary2Integers = {word: i for i, word in enumerate(sortedVocabulary, 1)}

    showItems(vocabulary2Integers)

    for review in reviews:
        
        reviews_ints.append([vocabulary2Integers[word] for word in review.split()])
        
    # Check the index of the first review
    print(reviews_ints[: 1])

    print(len(reviews_ints))

    # Positive is encoded to 1, nagative to 0
    labels = labels.split("\n")
    labels = np.array([1 if label == "positive" else 0 for label in labels])

    print(labels[: 10])

    # Filter the reviews of which the length of review is 0 (No propriate review), and then form an array
    nonzeroReviewsIndexs = [i for i, review in enumerate(reviews_ints) if len(review) > 0]

    print(len(nonzeroReviewsIndexs))

    # Filter the reviews in reviews_ints which string length is 0 via nonzeroReviewsIndexs
    reviews_ints = [reviews_ints[i] for i in nonzeroReviewsIndexs]
    # Filter the labels for above ones
    labels = np.array([labels[i] for i in nonzeroReviewsIndexs])

    print(len(reviews_ints))
    
    return reviews_ints, labels


def FeatureVectorize(reviews_ints):

    # Maximum length of review is 200
    length = 200

    # Create a new vector of features
    features = np.zeros((len(reviews_ints), length), dtype = int)

    # Truncate te string of review to 200, and fill out the features, will be itself if the case the string is less than 200
    for i, row in enumerate(reviews_ints):
        
        # Add heading 0s if less than 200
        features[i, -len(row):] = np.array(row)[: length]
        
    return features

reviews, labels = LoadReviews()
reviews, string, words = Preprocess(reviews)
reviews_ints, labels = Filter(reviews, labels)
features = FeatureVectorize(reviews_ints)

print(features)
print(features.shape)