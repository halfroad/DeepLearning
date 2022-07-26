import tensorflow as tf

def Prepare():
    
    mnist = tf.keras.datasets.mnist
    
    # import the dataset
    (imagesTrain, labelsTrain), (imagesTest, labelsTest) = mnist.load_data()
    
    return imagesTrain, labelsTrain, imagesTest, labelsTest

def Preprocess(imagesTrain, imagesTest):
    
    # Reshape
    imagesTrain4D = imagesTrain.reshape(imagesTrain.shape[0], 28, 28, 1).astype("float32")
    imagesTest4D = imagesTest.reshape(imagesTest.shape[0], 28, 28, 1).astype("float32")
    
    # Pixel standardization
    imagesTrain, imagesTest = imagesTrain4D / 225.0, imagesTest4D / 225.0
    
    return imagesTrain, imagesTest

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
    

imagesTrain, labelsTrain, imagesTest, labelsTest = Prepare()
imagesTrain, imagesTest = Preprocess(imagesTrain, imagesTest)

print("imagesTrain.shape = {}, labelsTrain.shape = {}, imagesTest.shape = {}, labelsTest.shape = {}".format(imagesTrain.shape, labelsTrain.shape, imagesTest.shape, labelsTest.shape))

TrainModel(imagesTrain, labelsTrain)
