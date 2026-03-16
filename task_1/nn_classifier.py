import numpy as np
from keras import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Input
from keras.callbacks import EarlyStopping

from interface import MnistClassifierInterface


class NNMnistClassifier(MnistClassifierInterface):
    def __init__(self):
        self.model = self._build_net()

        # Stop training when val_loss stops improving to prevent overfitting
        self.early_stop = EarlyStopping(
            monitor="val_loss", mode="min", patience=10, restore_best_weights=True
        )

    # Normalize pixel values to [0, 1] and flatten 28x28 images into 1D vectors
    def _preprocess(self, X):
        X_flat = X.astype("float32") / 255.0
        return X_flat.reshape(X_flat.shape[0], -1)

    def _build_net(self):
        model = Sequential(
            [
                Input(shape = (784,)),

                # First hidden layer
                Dense(128, activation="relu"),
                BatchNormalization(), #Standardize inputs
                Dropout(0.25), #Prevent overfitting by randomly dropping 25% of units
                
                # Second hidden layer
                Dense(64, activation="relu"),
                BatchNormalization(),
                Dropout(0.25),
                
                # Output layer with 10 units
                Dense(10, activation="softmax"),
            ]
        )

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy", # Suitable for integer-labeled classification
            metrics=["accuracy"]
        )

        return model

    def train(self, X_train, y_train):
        X_train_flat = self._preprocess(X_train)
        return self.model.fit(
            X_train_flat,
            y_train,
            epochs=50, # Number of complete passes through the training dataset
            validation_split=0.1, # Reserve 10% of data to monitor performance
            batch_size=128, # Number of samples processed before updating model weights
            callbacks=[self.early_stop],
        )

    def predict(self, X_test):
        X_test_flat = self._preprocess(X_test)
        y_pred = self.model.predict(X_test_flat)
        return np.argmax(y_pred, axis=1)
