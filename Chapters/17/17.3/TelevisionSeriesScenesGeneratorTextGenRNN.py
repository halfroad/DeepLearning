from textgenrnn import textgenrnn
import tensorflow as tf

def Generate():
    
    path = "../Exclusion/Datasets/SimpsonsEpisode18.txt"
    
    generator = textgenrnn()
        
    # Train the model from path
    generator.train_from_file(path, num_epochs = 30)
        
    # Generate the script by Temperature
    generator.generate()
    
Generate()