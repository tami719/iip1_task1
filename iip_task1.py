import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import cv2
import matplotlib.pyplot as plt

def plot_result(log):
    fig = plt.figure(figsize=(10, 3))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(log.history["accuracy"])
    ax1.plot(log.history["val_accuracy"])
    ax1.set_title("model2_acc")
    ax1.set_ylabel("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.legend(["Train", "Validation"], loc="best")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(log.history["loss"])
    ax2.plot(log.history["val_loss"])
    ax2.set_title("model2_loss")
    ax2.set_ylabel("Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend(["Train", "Validation"], loc="best")
    plt.show()



# get data
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

# normalization
X_train = X_train.astype(np.float32) / 255
X_test = X_test.astype(np.float32) / 255

# model1 (normal vgg16)
model1 = keras.Sequential(
    [
        layers.InputLayer(input_shape=(32, 32, 3)),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.Activation('relu'),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.Activation('relu'),
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.Activation('relu'),
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.Activation('relu'),
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(512, (3, 3), padding='same'),
        layers.Activation('relu'),
        layers.Conv2D(512, (3, 3), padding='same'),
        layers.Activation('relu'),
        layers.Conv2D(512, (3, 3), padding='same'),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),  
        layers.Conv2D(512, (3, 3), padding='same'),
        layers.Activation('relu'),
        layers.Conv2D(512, (3, 3), padding='same'),
        layers.Activation('relu'),
        layers.Conv2D(512, (3, 3), padding='same'),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),  

        layers.Flatten(),
        layers.Dense(4096, activation="relu"),
        layers.Dense(4096, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)
model1.summary()

# model2 (model1+batch_normalization)
model2 = keras.Sequential(
    [
        layers.InputLayer(input_shape=(32, 32, 3)),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(512, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(512, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(512, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),  
        layers.Conv2D(512, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(512, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(512, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),  

        layers.Flatten(),
        layers.Dense(4096, activation="relu"),
        layers.Dense(4096, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)
model2.summary()

model3 = keras.Sequential(
    [
        layers.InputLayer(input_shape=(32, 32, 3)),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(512, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(512, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(512, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),  
        layers.Conv2D(512, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(512, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(512, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),  

        layers.Flatten(),
        layers.Dense(4096, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(4096, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ]
)
model3.summary()

batch_size = 64
epochs = 20
lr = 1e-3
model1.compile(
        loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=lr), metrics=["accuracy"]
)
model2.compile(
        loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=lr), metrics=["accuracy"]
)
model3.compile(
        loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=lr), metrics=["accuracy"]
)


# learning
train_history = model1.fit(
    X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2
)

# plot the learning curve
plot_result(train_history)

# evaluate
loss_train, accuracy_train = model1.evaluate(X_test, y_test, verbose=0)
print("loss: ", loss_train)
print("accuracy: ", accuracy_train)