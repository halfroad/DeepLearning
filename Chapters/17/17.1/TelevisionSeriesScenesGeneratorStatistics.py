import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.python.client import device_lib
from wordcloud import WordCloud
from collections import Counter


def CheckGPUAvailability():
    
    # Check the availability of GPU
    gpus = tf.config.list_physical_devices("GPU")

    print(gpus)
    
def Preprocess(fileName, episodeId = 18):
    
    source = "../Exclusion/Datasets/simpsons_script_lines.csv"
    storeURL = os.path.dirname(source) + "/" + fileName
    
    if os.path.exists(storeURL):
        
        return storeURL
    
    else:
        
        # Read all episodes
        df = pd.read_csv(source, on_bad_lines = "skip", low_memory = False)
        
        # Acquire all the ids and texts from episodes
        df1 = df[["episode_id", "raw_text"]]
        # Use episode 18
        df2 = df1.query("episode_id == {}".format(episodeId))
        
        # Grab all the conversations from episode 18
        rawTexts = df2["raw_text"]
        
        # Iterate all the conversations from episode 18
        conversations = []
        
        for i, t in enumerate(rawTexts):
            
            # Ignore top 5 lines
            
            if i < 5:

                continue
            
            # Seperate by :, 1st element is the Speaker, the 2nd is the conversation content
            tupleConversation = t.split(":")
            
            # Replace the SPACE with _, and delete the " from head and tail
            speaker = tupleConversation[0].strip("\"").replace(" ", "_") + ":" + tupleConversation[1].strip("\"")
            
            # Use 2 \n for the new secene
            if speaker.startswith("(") and i > 5:
                
                speaker = "\n\n" + speaker
                
            conversations.append(speaker)
            
        with open(storeURL, "w") as file:
            
            for item in conversations:
                
                file.write("{}\n".format(item))
                
        return storeURL
        
def AcquireEpisode(storeURL):
    
    '''

    Acquire the stored episode
    
    '''
        
    with open(storeURL, "r") as f:
        
        episode = f.read()
        
        return episode
    
def StatisticsReport(fileName):
    
    episode = AcquireEpisode(fileName)
    
    print("Episode \"{}\" Statistics".format(fileName))
    print("Number of words gross statistics: {}".format(len({word: None for word in episode.split()})))
    
    print("\r")
    
    scenes = episode.split("\n\n")
    
    print("Number of scenes: {}".format(len(scenes)))
    
    sentencesPerScene = [scene.count("\n") for scene in scenes]
    
    print("Number of sentences per scene: {}".format(len(sentencesPerScene)))
    
    print("\r")
    
    sentences = [sentence for scene in scenes for sentence in scene.split("\n")]
    
    print("Number of sentences: {}".format(len(sentences)))
    
    wordsPerSentence = [len(sentence.split()) for sentence in sentences]
    
    print("Number of words per sentence in average: {}".format(np.average(wordsPerSentence)))
    
    print("\r")
    
    print("Top 5 Sentences")
    
    print("\n".join(episode.split("\n")[: 5]))
    
    return sentences
    
def DrawWordCloudImage(body = None, path = None, backgroudColor = "white", minimumFontSize = 5, maximumFontSize = 100, width = 700, height = 500, colorsMap = "Blues"):
    
    '''

    Draw the WordCloud image
    Parameter 0 body: text to draw
    Parameter 1 path: path of text
    Parameter 2 backgroundColor: backgroud color
    Parameter 3 minimumFontSize: Minimum Font Size
    Parameter 4 maximumFontSize: Maximum Font Size
    Parameter 5 widht: widht of window
    Parameter 6 height: height of window
    Parameter 7 colorsMap: Colors Map
    
    '''
    
    text = body
    
    if path is not None:
        
        # Read the text from path
        with open(path) as f:
            
            text = f.read()
            
    if len(text) > 0:
        
        # Generate the object of WordCloud
        wordCloud = WordCloud(background_color = backgroudColor,
                              min_font_size = minimumFontSize,
                              max_font_size = maximumFontSize,
                              width = width,
                              height = height,
                              colormap = colorsMap).generate(text)
        
        # Plot the figure
        plt.figure()
        plt.imshow(wordCloud, interpolation = "bilinear")
        
        # Whether to draw with grid
        plt.grid()
        
        # Hide ruler on the axis x and y
        plt.axis("off")
        
        # Show the figure
        plt.show()
        
def AcquireSpeakersConversations(sentences):
    
    speakers = []
    bodies = []
    
    for i, line in enumerate(sentences):
        
        # Seperate by :
        array = line.split(":")
        
        # Ignore those lines whose length is less than 2
        if len(array) == 2:
            
            # 1st element is Speaker,
            name = array[0]
            body = array[1]
            
            if len(name) > 0 and not name.startswith("("):
                
                speakers.append(name)
                
            if len(body) > 0 and not body.endswith(")"):
                
                bodies.append(body)
                
    return speakers, bodies

def DrawBar(speakers):
    
    # Count the speaker with Counter
    speakersCounter = Counter(speakers)
    
    # Print the most common words. Typically, the order by descending. Default is all the speakers
    mostCommon = speakersCounter.most_common()
    
    print("Most Common = {}".format(mostCommon))
    
    speakersVertical = [counter[0] for counter in mostCommon][: 15]
    speakersHorizontal = [counter[1] for counter in mostCommon][: 15]
    
    # Draw the barplot
    # Paramter orient: h == horizontal, v == vertical
    sns.barplot(x = speakersHorizontal, y = speakersVertical, orient = "h")
    
    plt.show()

fileName = "SimpsonsEpisode18.txt"

storeURL = Preprocess(fileName)
sentences = StatisticsReport(storeURL)

# DrawWordCloudImage(path = storeURL, backgroudColor = "black")

speakers, bodies = AcquireSpeakersConversations(sentences)

string = " ".join(speakers)

DrawWordCloudImage(body = string, backgroudColor = "black")

string = " ".join(bodies)

DrawWordCloudImage(body = string, backgroudColor = "black")

DrawBar(speakers)