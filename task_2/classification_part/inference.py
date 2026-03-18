import os
import json
import argparse

import numpy as np
from keras.models import load_model
from keras.utils import load_img, img_to_array
from keras.applications.resnet50 import preprocess_input

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.dirname(BASE_DIR)


def predict_image(image_path, model_path):
    # Load the trained model and predict class for a single image
    with open(os.path.join(TASK_DIR, "models", "classes.json"), "r") as f:
        classes = json.load(f)

    model = load_model(model_path)

    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # Apply ResNet50 specific preprocessing
    img_array = preprocess_input(img_array)

    y_pred = model.predict(img_array, verbose=0)

    # Get the index of the class with highest probability
    class_idx = np.argmax(y_pred[0])
    return classes[class_idx]


def get_args():
    parser = argparse.ArgumentParser(description="Predict Animal Class")

    parser.add_argument(
        "--image_dir",
        type=str,
        default=os.path.join(TASK_DIR, "demo", "image.png"),
        help="Path to image file",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.path.join(TASK_DIR, "models", "classification_model.keras"),
        help="Path to model",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    IMAGE_DIR = args.image_dir
    MODEL_DIR = args.model_dir

    animal = predict_image(args.image_dir, args.model_dir)
    print(f"Predicted class: {animal}")
