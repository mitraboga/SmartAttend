import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from src.config import ARTIFACTS_DIR, FACE_LABELS_PATH, FACE_MODEL_PATH, FACES_DIR, IMAGE_SIZE
from src.utils import ensure_directories, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the face recognition CNN.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_model(num_classes: int):
    import tensorflow as tf

    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
        ]
    )

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3)),
            data_augmentation,
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history, output_path: Path) -> None:
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.title("Face Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title("Face Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()

    if not FACES_DIR.exists():
        raise FileNotFoundError("Face dataset directory does not exist. Collect samples in data/faces first.")

    import tensorflow as tf

    train_ds = tf.keras.utils.image_dataset_from_directory(
        FACES_DIR,
        validation_split=args.validation_split,
        subset="training",
        seed=args.seed,
        image_size=IMAGE_SIZE,
        batch_size=args.batch_size,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        FACES_DIR,
        validation_split=args.validation_split,
        subset="validation",
        seed=args.seed,
        image_size=IMAGE_SIZE,
        batch_size=args.batch_size,
    )

    class_names = list(train_ds.class_names)
    if len(class_names) < 2:
        raise ValueError("At least two student classes are required to train the recognition model.")

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=autotune)
    val_ds = val_ds.prefetch(buffer_size=autotune)

    model = build_model(num_classes=len(class_names))
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

    ensure_directories(FACE_MODEL_PATH.parent, ARTIFACTS_DIR)
    model.save(FACE_MODEL_PATH)
    save_json(FACE_LABELS_PATH, class_names)
    plot_history(history, ARTIFACTS_DIR / "face_training_history.png")

    print(f"Saved face model to: {FACE_MODEL_PATH}")
    print(f"Saved class labels to: {FACE_LABELS_PATH}")


if __name__ == "__main__":
    main()
