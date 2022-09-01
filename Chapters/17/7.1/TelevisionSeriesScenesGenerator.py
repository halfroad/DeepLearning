import tensorflow as tf

mnist = tf.keras.datasets.mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train, X_test = X_train / 255.0, X_test / 255.0

def CreateModel():
    
    return tf.keras.models.Sequential([
        
        tf.keras.layers.Flatten(input_shape = (28, 28)),
        tf.keras.layers.Dense(512, activation = "relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation = "softmax")
        ])

model = CreateModel()

print(model.summary())

model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

tensorboardCallback = tf.keras.callbacks.TensorBoard(log_dir = "FitLog/", histogram_freq = 1)

model.fit(x = X_train, y = y_train, epochs = 5, validation_data = (X_test, y_test), callbacks = [tensorboardCallback])