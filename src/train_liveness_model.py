import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config import ARTIFACTS_DIR, IMAGE_SIZE, LIVENESS_DIR, LIVENESS_METADATA_PATH, LIVENESS_MODEL_PATH
from src.utils import ensure_directories, list_image_files, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the liveness detection CNN.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_model():
    import tensorflow as tf

    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.05),
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
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history, output_path: Path) -> None:
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.title("Liveness Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title("Liveness Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def find_best_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_accuracy = -1.0

    for threshold in np.linspace(0.3, 0.7, 81):
        predictions = (scores >= threshold).astype(int)
        accuracy = float(np.mean(predictions == labels))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(threshold)

    return best_threshold, best_accuracy


def train_liveness_model(
    *,
    epochs: int = 15,
    batch_size: int = 16,
    validation_split: float = 0.2,
    seed: int = 42,
) -> dict:
    if not LIVENESS_DIR.exists():
        raise FileNotFoundError("Liveness dataset directory does not exist. Populate data/liveness first.")

    real_count = len(list_image_files(LIVENESS_DIR / "real"))
    fake_count = len(list_image_files(LIVENESS_DIR / "fake"))
    if real_count < 2 or fake_count < 2:
        raise ValueError("At least two real and two fake liveness samples are required to train the model.")

    import tensorflow as tf

    train_ds = tf.keras.utils.image_dataset_from_directory(
        LIVENESS_DIR,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        label_mode="binary",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        LIVENESS_DIR,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        label_mode="binary",
    )

    class_names = list(train_ds.class_names)
    if sorted(class_names) != ["fake", "real"]:
        print(f"Warning: expected classes ['fake', 'real'], found {class_names}")

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=autotune)
    val_ds = val_ds.prefetch(buffer_size=autotune)

    model = build_model()
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    val_scores = model.predict(val_ds, verbose=0).flatten()
    val_labels = np.concatenate([labels.numpy().astype(int).flatten() for _, labels in val_ds], axis=0)
    threshold, threshold_accuracy = find_best_threshold(val_scores, val_labels)

    ensure_directories(LIVENESS_MODEL_PATH.parent, ARTIFACTS_DIR)
    model.save(LIVENESS_MODEL_PATH)
    plot_history(history, ARTIFACTS_DIR / "liveness_training_history.png")
    save_json(
        LIVENESS_METADATA_PATH,
        {
            "threshold": threshold,
            "threshold_accuracy": threshold_accuracy,
            "real_samples": real_count,
            "fake_samples": fake_count,
            "epochs": epochs,
            "class_names": class_names,
        },
    )

    return {
        "model_path": str(LIVENESS_MODEL_PATH),
        "metadata_path": str(LIVENESS_METADATA_PATH),
        "real_samples": real_count,
        "fake_samples": fake_count,
        "epochs": epochs,
        "class_names": class_names,
        "history_keys": list(history.history.keys()),
        "threshold": threshold,
        "threshold_accuracy": threshold_accuracy,
    }


def main() -> None:
    args = parse_args()
    result = train_liveness_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        seed=args.seed,
    )
    print(result)


if __name__ == "__main__":
    main()
