import nltk
from SpamEmailDetectorPreparation import Prepare

def ProcessByNaturalLanguageToolkit():

    nltk.download("punkt")
    
    df = Prepare()
    
    df["tokens"] = df["text"].map(lambda text: nltk.tokenize.word_tokenize(text))
    
    print(df["tokens"][0])


ProcessByNaturalLanguageToolkit()