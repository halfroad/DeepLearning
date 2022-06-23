import nltk
from SpamEmailDetectorPreparation import Prepare
import shutil
import os
import string
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def ProcessByNaturalLanguageToolkit():

    #nltk.download("punkt")
    
    df = Prepare()
    
    name = "nltk_data"
    path = os.path.expanduser("~") + "/" + name
    
    if not os.path.isdir(path):
        shutil.copytree("../../" + name, path)
    
    df["tokens"] = df["text"].map(lambda text: nltk.tokenize.word_tokenize(text))
    
    print(df["tokens"])
    
    # nltk.download("stopwords")
    
    stopWords = nltk.corpus.stopwords.words("english")
    
    print("There are {} Stop Words, top 15 are: {}.".format(len(stopWords), stopWords[:15]))
    
    df["textStopWords"] = df["tokens"].map(lambda tokens: " ".join([t for t in tokens if t not in stopWords]))
    
    print(df["textStopWords"])
    
    print("string.punctuation are: {}.".format(string.punctuation))
    df["punctuationText"] = df["textStopWords"].map(lambda text: "".join([w for w in text if w not in string.punctuation]))
    
    print(df["punctuationText"])
    
    # Download wordnet to revert the original meaning
    #nltk.download("wordnet")
    
    lemmatizer = nltk.WordNetLemmatizer()
    
    df["lemmatizedText"] = df["punctuationText"].map(lambda text: lemmatizer.lemmatize(text))
    
    print(df["lemmatizedText"])
    
    return df


# A fucntion to plot the word cloud
def PlotWordCloud(words):
    
    # Create a object of word cloud, size of window is 300 * 300 inches
    wordCloud = WordCloud(width = 300, height = 300).generate(words)
    
    # Create a matplotlib object
    plt.figure(figsize = (8, 6), facecolor = "k")
    
    # Show the word cloud on matplotlib object
    plt.imshow(wordCloud)
    
    plt.show()
    
# Split the train and test set
def SplitTrainTest(df):
    
    features = df["lemmatizedText"]
    labels = df["spam"]
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.15)
    
    print("X_train.shape = {}, y_train.shape = {}".format(X_train.shape, y_train.shape))
    print("X_test.shape = {}, y_test.shape = {}".format(X_test.shape, y_test.shape))
    
    return X_train, X_test, y_train, y_test

    
df = ProcessByNaturalLanguageToolkit()

spamEmailWords = "".join(list(df[df["spam"] == 1]["lemmatizedText"]))

PlotWordCloud(spamEmailWords)

normalEmailWords = "".join(list(df[df["spam"] == 0]["lemmatizedText"]))

PlotWordCloud(normalEmailWords)

SplitTrainTest(df)