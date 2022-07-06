from keras.preprocessing.text import Tokenizer

texts = ['I love AI in China", "", "AI"]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

print("Tokenizer.word_index = {}.".format(tokenizer.word_index))
print("Tokenizer.texts_to_sequences = {}".format(tokenizer.texts_to_sequences(texts)))