import numpy as np
from keras import Sequential
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Dropout,
    Flatten,
    Dense,
    Input
)
from keras.callbacks import EarlyStopping

from interface import MnistClassifierInterface


class CNNMnistClassifier(MnistClassifierInterface):
    def __init__(self):
        self.model = self._build_net()

        # Stop training when val_loss stops improving to prevent overfitting
        self.early_stop = EarlyStopping(
            monitor="val_loss", mode="min", patience=10, 
            restore_best_weights=True
        )

    # Normalize pixel values to [0, 1] and add channel dimension for Conv2D
    def _preprocess(self, X):
        X_train_proc = X.astype("float32") / 255.0
        return X_train_proc[..., np.newaxis]

    def _build_net(self):
        model = Sequential([
            # Define the input shape matching MNIST images
            Input(shape=(28, 28, 1)),

            # First Conv block
            Conv2D(32, (3, 3), padding="same",
                   activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),  # Reduce spatial dimensions by half
            BatchNormalization(),  #Standardize inputs
            Dropout(0.1), #Prevent overfitting by randomly dropping 10% of units

            # Second Conv block
            Conv2D(64, (3, 3), padding="same", activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            BatchNormalization(),
            Dropout(0.1),

            # Transition from convolutional layers to fully connected layers
            Flatten(),
            Dense(128, activation="relu"),
            BatchNormalization(),
            Dropout(0.5),

            # Output layer with 10 units
            Dense(10, activation="softmax")
        ])

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy", # Suitable for integer-labeled classification
            metrics=["accuracy"]
        )

        return model

    def train(self, X_train, y_train):
        X_train_proc = self._preprocess(X_train)
        return self.model.fit(
            X_train_proc,
            y_train,
            epochs=15, # Number of complete passes through the training dataset
            validation_split=0.1, # Reserve 10% of data to monitor performance
            batch_size=128, # Number of samples processed before updating model weights
            callbacks=[self.early_stop],
        )

    def predict(self, X_test):
        X_test_proc = self._preprocess(X_test)
        y_pred = self.model.predict(X_test_proc)
        return np.argmax(y_pred, axis=1)
