from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.config import ARTIFACTS_DIR, FACE_LABELS_PATH, FACE_MODEL_PATH, FACES_DIR, IMAGE_SIZE, LIVENESS_DIR, LIVENESS_METADATA_PATH, LIVENESS_MODEL_PATH
from src.database import save_evaluation_report
from src.utils import ensure_directories, load_json, save_json


def _plot_confusion_matrix(matrix: np.ndarray, labels: list[str], title: str, output_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def evaluate_face_model() -> dict:
    if not FACE_MODEL_PATH.exists() or not FACE_LABELS_PATH.exists():
        raise FileNotFoundError("Face recognition model is missing.")

    import tensorflow as tf

    labels = load_json(FACE_LABELS_PATH, default=[])
    dataset = tf.keras.utils.image_dataset_from_directory(
        FACES_DIR,
        shuffle=False,
        image_size=IMAGE_SIZE,
        batch_size=16,
    )
    model = tf.keras.models.load_model(FACE_MODEL_PATH)

    y_true = np.concatenate([labels_batch.numpy() for _, labels_batch in dataset], axis=0)
    predictions = model.predict(dataset, verbose=0)
    y_pred = np.argmax(predictions, axis=1)

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True, zero_division=0)
    payload = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "labels": labels,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }

    ensure_directories(ARTIFACTS_DIR)
    _plot_confusion_matrix(matrix, labels, "Face Recognition Confusion Matrix", ARTIFACTS_DIR / "face_confusion_matrix.png")
    save_json(ARTIFACTS_DIR / "face_metrics.json", payload)
    save_evaluation_report("face_model", payload)
    return payload


def evaluate_liveness_model() -> dict:
    if not LIVENESS_MODEL_PATH.exists():
        raise FileNotFoundError("Liveness model is missing.")

    import tensorflow as tf

    dataset = tf.keras.utils.image_dataset_from_directory(
        LIVENESS_DIR,
        shuffle=False,
        image_size=IMAGE_SIZE,
        batch_size=16,
        label_mode="binary",
    )
    class_names = list(dataset.class_names)
    model = tf.keras.models.load_model(LIVENESS_MODEL_PATH)
    metadata = load_json(LIVENESS_METADATA_PATH, default={})
    threshold = float(metadata.get("threshold", 0.5))

    y_true = np.concatenate([labels_batch.numpy().astype(int).flatten() for _, labels_batch in dataset], axis=0)
    scores = model.predict(dataset, verbose=0).flatten()
    y_pred = (scores >= threshold).astype(int)

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    tn, fp, fn, tp = matrix.ravel()
    far = float(fp / (fp + tn)) if (fp + tn) else 0.0
    frr = float(fn / (fn + tp)) if (fn + tp) else 0.0

    payload = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "labels": class_names,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "false_acceptance_rate": far,
        "false_rejection_rate": frr,
        "threshold": threshold,
    }

    ensure_directories(ARTIFACTS_DIR)
    _plot_confusion_matrix(matrix, class_names, "Liveness Confusion Matrix", ARTIFACTS_DIR / "liveness_confusion_matrix.png")
    save_json(ARTIFACTS_DIR / "liveness_metrics.json", payload)
    save_evaluation_report("liveness_model", payload)
    return payload


def run_all_evaluations() -> dict:
    results: dict[str, dict] = {}

    try:
        results["face_model"] = evaluate_face_model()
    except FileNotFoundError as error:
        results["face_model"] = {"error": str(error)}

    try:
        results["liveness_model"] = evaluate_liveness_model()
    except FileNotFoundError as error:
        results["liveness_model"] = {"error": str(error)}

    ensure_directories(ARTIFACTS_DIR)
    save_json(ARTIFACTS_DIR / "evaluation_summary.json", results)
    return results


def main() -> None:
    results = run_all_evaluations()
    print(results)


if __name__ == "__main__":
    main()
