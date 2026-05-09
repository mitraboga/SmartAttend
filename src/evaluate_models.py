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
    matrix = np.asarray(matrix)
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_totals, out=np.zeros_like(matrix, dtype=float), where=row_totals != 0)
    annotations = np.empty_like(matrix, dtype=object)
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            annotations[row_index, col_index] = f"{matrix[row_index, col_index]}\n{normalized[row_index, col_index] * 100:.1f}%"

    plt.figure(figsize=(8.4, 6.4))
    sns.heatmap(
        normalized,
        annot=annotations,
        fmt="",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": "Row-normalized proportion"},
        xticklabels=[label.title() for label in labels],
        yticklabels=[label.title() for label in labels],
    )
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
    plt.figure(figsize=(8.4, 6.4))
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


def _select_operating_threshold(
    fpr: np.ndarray,
    tpr: np.ndarray,
    thresholds: np.ndarray,
    default_threshold: float,
) -> tuple[float, tuple[float, float]]:
    finite_mask = np.isfinite(thresholds)
    if not np.any(finite_mask):
        return default_threshold, (0.0, 0.0)

    candidate_fpr = fpr[finite_mask]
    candidate_tpr = tpr[finite_mask]
    candidate_thresholds = thresholds[finite_mask]
    scores = candidate_tpr - candidate_fpr
    best_score = float(np.max(scores))
    best_indices = np.where(np.isclose(scores, best_score))[0]

    chosen_index = int(best_indices[0])
    if len(best_indices) > 1:
        best_thresholds = candidate_thresholds[best_indices]
        chosen_index = int(best_indices[np.argmax(best_thresholds)])

    chosen_threshold = float(candidate_thresholds[chosen_index])
    chosen_point = (float(candidate_fpr[chosen_index]), float(candidate_tpr[chosen_index]))
    return chosen_threshold, chosen_point


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
    original_threshold = float(metadata.get("threshold", 0.5))
    images, y_true = _load_labeled_images(LIVENESS_DIR, class_names)

    scores = model.predict(images, verbose=0).flatten()
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    threshold, operating_point = _select_operating_threshold(fpr, tpr, thresholds, original_threshold)
    y_pred = (scores >= threshold).astype(int)

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    tn, fp, fn, tp = matrix.ravel()
    far = float(fp / (fp + tn)) if (fp + tn) else 0.0
    frr = float(fn / (fn + tp)) if (fn + tp) else 0.0
    roc_auc = float(auc(fpr, tpr))
    evaluation_dir = ARTIFACTS_DIR / "evaluation"
    metadata["threshold"] = threshold
    metadata["threshold_accuracy"] = float(accuracy_score(y_true, y_pred))
    metadata["recalibrated_from_threshold"] = original_threshold
    metadata["roc_auc"] = roc_auc
    save_json(LIVENESS_METADATA_PATH, metadata)

    payload = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "labels": class_names,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "false_acceptance_rate": far,
        "false_rejection_rate": frr,
        "threshold": threshold,
        "original_threshold": original_threshold,
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
    _plot_confusion_matrix(matrix, class_names, "Liveness Confusion Matrix", evaluation_dir / "liveness_confusion_matrix.png")
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
