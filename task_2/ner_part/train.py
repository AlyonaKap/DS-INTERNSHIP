import os
import json
import argparse

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForTokenClassification

MODEL_NAME = "distilbert-base-uncased"
LABEL_MAP = {"O": 0, "ANIMAL": 1}
MAX_LEN = 128
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.dirname(BASE_DIR)


def tokenize_and_label(raw_data, tokenizer):
    # Convert text and annotations into model inputs
    all_input_ids = []
    all_attention_masks = []
    all_labels = []

    for entry in raw_data:
        text = entry["data"]["text"]
        annotations = entry["annotations"][0]["result"]

        # Tokenize text and get character offsets
        encoding = tokenizer(
            text,
            return_offsets_mapping=True,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )

        # Initialize labels with 'O' for all tokens
        labels = torch.zeros(MAX_LEN, dtype=torch.long)
        offsets = encoding["offset_mapping"].squeeze().tolist()

        # Map character-level annotations to token-level labels
        for anno in annotations:
            start, end = anno["value"]["start"], anno["value"]["end"]
            for i, (o_start, o_end) in enumerate(offsets):
                if o_start == o_end:
                    continue
                if o_start >= start and o_end <= end:
                    labels[i] = 1

        all_input_ids.append(encoding["input_ids"].squeeze())
        all_attention_masks.append(encoding["attention_mask"].squeeze())
        all_labels.append(labels)

    return TensorDataset(
        torch.stack(all_input_ids),
        torch.stack(all_attention_masks),
        torch.stack(all_labels),
    )


def build_model():
    # Load pre-trained DistilBert with a token classification head
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABEL_MAP)
    )
    return model


def train_model(model, loader, device):
    # Fine-tune the model on the labeled dataset
    optimizer = AdamW(model.parameters(), lr=5e-5)
    model.train()

    for epoch in range(EPOCHS):
        for input_ids, attention_mask, labels in loader:
            optimizer.zero_grad()

            outputs = model(
                input_ids.to(device),
                attention_mask=attention_mask.to(device),
                labels=labels.to(device),
            )

            # Compute gradients and update model weights
            outputs.loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {outputs.loss.item():.4f}")

    return model


def get_args():
    parser = argparse.ArgumentParser(description="Train NER Model")

    parser.add_argument(
        "--data_path",
        type=str,
        default=os.path.join(TASK_DIR, "data", "labeled_animals.json"),
        help="Path to labeled data",
    )
    parser.add_argument(
        "--batch_size", type=int, default=5, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of epochs"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    EPOCHS = args.epochs

    # Select GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.data_path, "r") as f:
        raw_data = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = tokenize_and_label(raw_data, tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = build_model()
    model.to(device)
    model = train_model(model, loader, device)

    # Save the fine-tuned model, tokenizer, and label map
    model.save_pretrained(os.path.join(TASK_DIR, "models", "ner_model"))
    tokenizer.save_pretrained(os.path.join(TASK_DIR, "models", "ner_model"))

    with open(os.path.join(TASK_DIR, "models", "label_map.json"), "w") as f:
        json.dump(LABEL_MAP, f)
