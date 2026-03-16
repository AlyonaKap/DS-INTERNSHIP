# Task 1: MNIST Classification (RF, FNN, CNN)

## Overview

This project implements three classification models for the MNIST handwritten digit dataset. Each model is built as a separate class behind a unified interface, allowing seamless switching between algorithms without changing the calling code.

## Project Structure

- `interface.py`: Abstract base class `MnistClassifierInterface` with two abstract methods: `train` and `predict` 
- `rf_classifier.py`: `RFMnistClassifier` that is Random Forest implementation 
- `nn_classifier.py`: `NNMnistClassifier` that is Feed-Forward Neural Network implementation 
- `cnn_classifier.py`: `CNNMnistClassifier` that is Convolutional Neural Network implementation 
- `classifier.py`: `MnistClassifier` that is facade class that selects the appropriate model based on the `algorithm` parameter 
- `solution.ipynb`: Jupyter notebook with exploratory data analysis, training, evaluation and edge case analysis 

## Models

- **Random Forest (`rf`)**: classical ensemble method from scikit-learn. Images are flattened into 1D vectors and normalized to [0, 1]

- **Feed-Forward Neural Network (`nn`)**: fully connected Keras network with two hidden layers (128 and 64 units), batch normalization, dropout and EarlyStopping callback to prevent overfitting

- **Convolutional Neural Network (`cnn`)**: Keras CNN with two convolutional blocks (32 and 64 filters), max pooling, batch normalization, dropout and a dense classification head. Operates on 2D image data with a channel dimension

## Setup and Installation

1. Clone the repository

```
git clone https://github.com/AlyonaKap/DS-INTERNSHIP.git
cd task_1
```

2. Create and activate a virtual environment (Recommended)

Linux 
```
python3 -m venv venv
source venv/bin/activate
```

Windows 
```
python -m venv venv
venv\Scripts\Activate.ps1
```
3. Install dependencies

```
pip install -r requirements.txt
```

4. Run the task

You can open and run cells in the provided Jupyter notebook:

```
jupyter notebook solution.ipynb
```

Alternatively, you can create your own Python script and import the classifier directly:

```python
from keras.datasets import mnist
from classifier import MnistClassifier

(X_train, y_train), (X_test, y_test) = mnist.load_data()

clf = MnistClassifier(algorithm="cnn")  # options: "rf", "nn", "cnn"
clf.train(X_train, y_train)
predictions = clf.predict(X_test)
```

## Results

| Model | Accuracy |
|-------|----------|
| Random Forest | 97.04% |
| Feed-Forward NN | 97.99% |
| CNN | 99.25% |

The CNN achieves the highest accuracy by leveraging spatial feature extraction through convolutional layers, while the FNN and Random Forest provide strong baselines with simpler architectures.
