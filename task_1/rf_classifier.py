from sklearn.ensemble import RandomForestClassifier

from interface import MnistClassifierInterface


class RFMnistClassifier(MnistClassifierInterface):
    def __init__(self):
        self.model = RandomForestClassifier(random_state=42)

    # Normalize pixel values to [0, 1] and flatten 28x28 images into 1D vectors
    def _preprocess(self, X):
        X_flat = X.astype("float32") / 255.0
        return X_flat.reshape(X_flat.shape[0], -1)

    def train(self, X_train, y_train):
        X_train_flat = self._preprocess(X_train)
        self.model.fit(X_train_flat, y_train)

    def predict(self, X_test):
        X_test_flat = self._preprocess(X_test)
        return self.model.predict(X_test_flat)
