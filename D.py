import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from keras import regularizers

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),

        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same'),
        layers.Dropout(0.15),
        layers.MaxPooling2D(pool_size=(2, 2), padding='same'),


        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same'),
        layers.Dropout(0.25),
        layers.MaxPooling2D(pool_size=(2, 2), padding='same'),
        
#         layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same'),
#         layers.Dropout(0.25),
#         layers.MaxPooling2D(pool_size=(2, 2), padding='same'),
#         layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same'),
#         layers.Dropout(0.25),
#         layers.MaxPooling2D(pool_size=(2, 2), padding='same'),
#         layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same'),
#         layers.Dropout(0.25),
#         layers.MaxPooling2D(pool_size=(2, 2), padding='same'),
#         layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same'),
#         layers.Dropout(0.25),
#         layers.MaxPooling2D(pool_size=(2, 2), padding='same'),
#         layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same'),
#         layers.Dropout(0.25),
#         layers.MaxPooling2D(pool_size=(2, 2), padding='same'),


        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same'),
        layers.Dropout(0.15),
        layers.MaxPooling2D(pool_size=(2, 2), padding='same'),


        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same'),
        layers.Dropout(0.15),
        layers.MaxPooling2D(pool_size=(2, 2), padding='same'),

        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same'),
        layers.Dropout(0.15),
        layers.MaxPooling2D(pool_size=(2, 2), padding='same'),

        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same'),
        layers.Dropout(0.15),
        layers.MaxPooling2D(pool_size=(2, 2), padding='same'),

        layers.Flatten(),

        layers.Dropout(0.30),
        layers.Dense(100, activation='ReLU', kernel_regularizer=regularizers.l1_l2(l1=0.00001, l2=0.0001)),
        
        layers.Dense(100, activation='ReLU', kernel_regularizer=regularizers.l1_l2(l1=0.00001, l2=0.0001)),

        layers.Dropout(0.35),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

batch_size = 128
epochs = 20

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
