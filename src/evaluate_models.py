from pathlib import Path

import cv2
import matplotlib
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, roc_curve

from src.config import ARTIFACTS_DIR, FACE_LABELS_PATH, FACE_MODEL_PATH, FACES_DIR, IMAGE_SIZE, LIVENESS_DIR, LIVENESS_METADATA_PATH, LIVENESS_MODEL_PATH
from src.database import save_evaluation_report
from src.utils import ensure_directories, list_image_files, load_json, resize_and_normalize, save_json

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _plot_confusion_matrix(matrix: np.ndarray, labels: list[str], title: str, output_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float,
    output_path: Path,
    *,
    operating_point: tuple[float, float] | None = None,
) -> None:
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="#0f766e", linewidth=2.2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#94a3b8", linestyle="--", linewidth=1.4, label="Random baseline")
    if operating_point is not None:
        plt.scatter(
            [operating_point[0]],
            [operating_point[1]],
            color="#c95d3d",
            s=48,
            zorder=3,
            label="Operating threshold",
        )
    plt.title("Liveness Model ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.02)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.18)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _load_labeled_images(root_dir: Path, class_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    images: list[np.ndarray] = []
    labels: list[int] = []

    for label_index, class_name in enumerate(class_names):
        class_dir = root_dir / class_name
        if not class_dir.exists():
            continue
        for image_path in list_image_files(class_dir):
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                continue
            images.append(resize_and_normalize(image_bgr, IMAGE_SIZE))
            labels.append(label_index)

    if not images:
        raise FileNotFoundError(f"No labeled images found under {root_dir}.")

    return np.asarray(images, dtype="float32"), np.asarray(labels, dtype=int)


def evaluate_face_model() -> dict:
    if not FACE_MODEL_PATH.exists() or not FACE_LABELS_PATH.exists():
        raise FileNotFoundError("Face recognition model is missing.")

    import tensorflow as tf

    labels = load_json(FACE_LABELS_PATH, default=[])
    images, y_true = _load_labeled_images(FACES_DIR, labels)
    model = tf.keras.models.load_model(FACE_MODEL_PATH)

    predictions = model.predict(images, verbose=0)
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

    model = tf.keras.models.load_model(LIVENESS_MODEL_PATH)
    metadata = load_json(LIVENESS_METADATA_PATH, default={})
    class_names = list(metadata.get("class_names", [])) or sorted(path.name for path in LIVENESS_DIR.iterdir() if path.is_dir())
    threshold = float(metadata.get("threshold", 0.5))
    images, y_true = _load_labeled_images(LIVENESS_DIR, class_names)

    scores = model.predict(images, verbose=0).flatten()
    y_pred = (scores >= threshold).astype(int)

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    tn, fp, fn, tp = matrix.ravel()
    far = float(fp / (fp + tn)) if (fp + tn) else 0.0
    frr = float(fn / (fn + tp)) if (fn + tp) else 0.0
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = float(auc(fpr, tpr))
    operating_index = int(np.argmin(np.abs(thresholds - threshold)))
    operating_point = (float(fpr[operating_index]), float(tpr[operating_index]))
    evaluation_dir = ARTIFACTS_DIR / "evaluation"

    payload = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "labels": class_names,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "false_acceptance_rate": far,
        "false_rejection_rate": frr,
        "threshold": threshold,
        "roc_auc": roc_auc,
        "roc_curve": {
            "false_positive_rate": fpr.tolist(),
            "true_positive_rate": tpr.tolist(),
            "thresholds": thresholds.tolist(),
            "operating_point": {
                "fpr": operating_point[0],
                "tpr": operating_point[1],
            },
        },
    }

    ensure_directories(ARTIFACTS_DIR, evaluation_dir)
    _plot_confusion_matrix(matrix, class_names, "Liveness Confusion Matrix", ARTIFACTS_DIR / "liveness_confusion_matrix.png")
    _plot_roc_curve(fpr, tpr, roc_auc, evaluation_dir / "roc_curve.png", operating_point=operating_point)
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
