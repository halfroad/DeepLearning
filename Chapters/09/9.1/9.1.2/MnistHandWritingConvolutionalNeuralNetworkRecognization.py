import tensorflow as tf

from MnistHandWritingConvolutionalPreparation import Prepare

def CreateModel():
    
    model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(filters = 16, kernel_size = (5, 5), padding = "same", input_shape = (28, 28, 1), activation = "relu"),
                                        tf.keras.layers.MaxPooling2D(pool_size= (2, 2)),
                                        tf.keras.layers.Conv2D(filters = 36, kernel_size = (5, 5), padding = "same", activation = "relu"),
                                        tf.keras.layers.MaxPooling2D(pool_size = (2, 2)),
                                        tf.keras.layers.Dropout(0.25),
                                        tf.keras.layers.Flatten(),
                                        tf.keras.layers.Dense(128, activation = "relu"),
                                        tf.keras.layers.Dropout(0.5),
                                        tf.keras.layers.Dense(10, activation = "softmax")
                                        ])
    
    return model

def TrainModel(imagesTrain, lablesTrain):
    
    model = CreateModel()
    
    print(model.summary())
    
    # Configure the model
    model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
    
    # Train the model
    model.fit(x = imagesTrain, y = lablesTrain, validation_split = 0.2, epochs = 20, batch_size = 300, verbose = 2)
    

imagesTrain, labelsTrain, imagesTest, labelsTest, imagesValidation, labelsValidation = Prepare()

print("imagesTrain.shape = {}, labelsTrain.shape = {}, imagesTest.shape = {}, labelsTest.shape = {}, imagesValidation = {}, labelsValidation = {}".format(imagesTrain.shape, labelsTrain.shape, imagesTest.shape, labelsTest.shape, imagesValidation.shape, labelsValidation.shape))

#TrainModel(imagesTrain, labelsTrain)
