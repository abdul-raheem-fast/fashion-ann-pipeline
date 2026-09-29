import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def evaluate():
    processed_dir = os.path.join("data", "processed")
    model_path = os.path.join("models", "model.h5")
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print("Loading test data and trained model...")
    x_test = np.load(os.path.join(processed_dir, "x_test.npy"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))
    model = tf.keras.models.load_model(model_path)

    print("Evaluating model performance on test set...")
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f} | Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Generate predictions and confusion matrix
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(9, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Fashion-MNIST Confusion Matrix (Accuracy: {accuracy*100:.2f}%)")
    plt.colorbar()
    tick_marks = np.arange(len(CLASS_NAMES))
    plt.xticks(tick_marks, CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(tick_marks, CLASS_NAMES)

    # Annotate numbers in confusion matrix cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.tight_layout()
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    cm_path = os.path.join("models", "confusion_matrix.png")
    cm_report_path = os.path.join(reports_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.savefig(cm_report_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix plot to '{cm_path}' and '{cm_report_path}'.")

    # Save metrics to root metrics.json
    metrics = {
        "test_loss": round(float(loss), 4),
        "test_accuracy": round(float(accuracy), 4)
    }

    metrics_file = "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved evaluation metrics to '{metrics_file}': {metrics}")

if __name__ == "__main__":
    evaluate()
