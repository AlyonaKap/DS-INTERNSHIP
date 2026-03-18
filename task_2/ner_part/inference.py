import os
import json
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, logging

logging.disable_progress_bar()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.dirname(BASE_DIR)


def predict_text(text, model_dir):
    # Load the fine-tuned model and extract animal entities from text
    with open(os.path.join(TASK_DIR, "models", "label_map.json"), "r") as f:
        label_map = json.load(f)

    id_to_label = {v: k for k, v in label_map.items()}

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # Disable gradient calculation for faster inference
    with torch.no_grad():
        outputs = model(**inputs)
        # Get the most likely class for each token
        predictions = torch.argmax(outputs.logits, dim=2)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    animals = []

    for token, pred in zip(tokens, predictions[0].tolist()):
        if id_to_label.get(pred) == "ANIMAL":
            animals.append(token)

    if not animals:
        return ""

    return animals[0]


def get_args():
    parser = argparse.ArgumentParser(description="Predict Animals from Text")

    parser.add_argument(
        "--text",
        type=str,
        default="I see duck in the garden ",
        help="Text to analyze",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.path.join(TASK_DIR, "models", "ner_model"),
        help="Path to model",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    animals = predict_text(args.text, args.model_dir)
    print(f"Animal Entity: {animals}")
