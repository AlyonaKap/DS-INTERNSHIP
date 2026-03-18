import os
import json
import argparse

from keras import Sequential, Model
from keras.utils import image_dataset_from_directory
from keras.layers import (
    Dropout,
    GlobalAveragePooling2D,
    Dense,
    Input,
    RandomFlip,
    RandomZoom,
    RandomRotation,
)
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.callbacks import EarlyStopping, ModelCheckpoint

IMAGE_SIZE = (224, 224)
CHANNELS = 3
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.dirname(BASE_DIR)


def set_preprocess(data_dir, subset_name=None, is_test=False):
    if is_test:
        return image_dataset_from_directory(
            directory=data_dir,
            label_mode="categorical",
            class_names=classes,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )
    else:
        return image_dataset_from_directory(
            directory=data_dir,
            validation_split=0.2,
            subset=subset_name,
            seed=42,
            label_mode="categorical",
            class_names=classes,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
        )


def build_base_model():
    # Load pre-trained ResNet50 
    base_model = ResNet50(
        input_shape=IMAGE_SIZE + (CHANNELS,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False # Freeze model weights
    return base_model


def augmentation():
    # Apply random transformations to prevent overfitting
    data_augmentation = Sequential([
        RandomFlip("horizontal"),
        RandomRotation(0.1),
        RandomZoom(0.1),
    ])
    return data_augmentation


def build_clf_model(base_model):
    inputs = Input(shape=IMAGE_SIZE + (CHANNELS,))

    x = augmentation()(inputs)
    x = preprocess_input(x)

    # Pass inputs through the frozen base model
    x = base_model(x, training=False)
    # Reduce spatial dimensions
    x = GlobalAveragePooling2D()(x)

    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)
    # Output layer for multi-class classification
    outputs = Dense(len(classes), activation="softmax")(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def callbacks_settings():
    callbacks = [
        # Save the model when performance improves
        ModelCheckpoint(
            os.path.join(TASK_DIR, "models", "classification_model.keras"),
            save_best_only=True,
        ),
        # Stop training when val_loss stops improving to prevent overfitting
        EarlyStopping(patience=10, restore_best_weights=True),
    ]
    return callbacks


def train_model():
    train_ds = set_preprocess(TRAIN_DIR, "training")
    val_ds = set_preprocess(TRAIN_DIR, "validation")
    callbacks = callbacks_settings()
    base_model = build_base_model()
    model = build_clf_model(base_model)
    return model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks
    )


def get_args():
    parser = argparse.ArgumentParser(description="Train Animal Detection Model")

    parser.add_argument(
        "--train_dir",
        type=str,
        default=os.path.join(TASK_DIR, "data", "train"),
        help="Path to training data",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default=os.path.join(TASK_DIR, "data", "test"),
        help="Path to test data",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    TRAIN_DIR = args.train_dir
    TEST_DIR = args.test_dir
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs

    classes = sorted(os.listdir(TRAIN_DIR))

    with open(os.path.join(TASK_DIR, "models", "classes.json"), "w") as f:
        json.dump(classes, f)

    history = train_model()
