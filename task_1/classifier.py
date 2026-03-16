from rf_classifier import RFMnistClassifier
from nn_classifier import NNMnistClassifier
from cnn_classifier import CNNMnistClassifier


class MnistClassifier:
    def __init__(self, algorithm):
        # Registry mapping algorithm keys to their corresponding classifier classes
        self.algorithms = {
            "rf": RFMnistClassifier,
            "nn": NNMnistClassifier,
            "cnn": CNNMnistClassifier,
        }

        if algorithm not in self.algorithms:
            raise ValueError(f"Please choose between: {list(self.algorithms.keys())}")

        self.model = self.algorithms[algorithm]()

    def train(self, x_train, y_train):
        return self.model.train(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)
