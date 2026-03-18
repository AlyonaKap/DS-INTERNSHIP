import os
import argparse

from classification_part.inference import predict_image
from ner_part.inference import predict_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def animals_pipeline(image_path, text):

    classification_model_path = os.path.join(BASE_DIR, "models", "classification_model.keras")
    ner_model_dir = os.path.join(BASE_DIR, "models", "ner_model")

    animal_image = predict_image(image_path, classification_model_path)
    animal_text = predict_text(text, ner_model_dir)

    match = False
    if animal_text and animal_image:
        match = (animal_text.lower() == animal_image.lower())

    return match

def get_args():
    parser = argparse.ArgumentParser(description="Animals Detetction Pipeline")
    parser.add_argument(
        "--image_dir",
        type=str,
        default=os.path.join(BASE_DIR, "demo", "tiger.png"),
        help="Path to image file",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="I see the pig",
        help="Text describing the image",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    result = animals_pipeline(args.image_dir, args.text)

    print(result)
