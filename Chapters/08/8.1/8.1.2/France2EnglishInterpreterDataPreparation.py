from os import path


def Prepare():
    
    # Load the criticisms
    url = path.relpath("../../../../MyBook/Chapter-8-Language-Translation/fra-eng/fra.txt")
    
    with open(url, "rt", encoding="utf-8") as f:
        
        text = f.read()
        
    # Top 80 characters
    print(text[: 80])
    
    lines = text.strip().split('\n')
    linePairs = [line.split('\t') for line in lines]
    
    print(linePairs[: 15])
    
    pairsLength = len(linePairs)
    englishPairsLengths = [len(linePair[0]) for linePair in linePairs]
    francePairsLengths = [len(linePair[1]) for linePair in linePairs]
    
    print("There are {} English-France pairs:".format(pairsLength))
    print("The length of shortest sentence in France is {}, the length of longest one is {}".format(min(francePairsLengths), max(francePairsLengths)))
    print("The length of shortest sentence in English is {}, the length of longest one is {}".format(min(englishPairsLengths), max(englishPairsLengths)))

          
    

Prepare()