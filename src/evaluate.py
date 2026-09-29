import os
import json
import numpy as np
import matplotlib.pyplot as plt

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def compute_confusion_matrix(y_true, y_pred, num_classes=10):
    try:
        from sklearn.metrics import confusion_matrix
        return confusion_matrix(y_true, y_pred)
    except ImportError:
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        return cm

def evaluate():
    processed_dir = os.path.join("data", "processed")
    model_path = os.path.join("models", "model.h5")
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print("Loading test data and trained model...")
    x_test = np.load(os.path.join(processed_dir, "x_test.npy"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))

    try:
        import tensorflow as tf
        print("Loading model via TensorFlow Keras...")
        model = tf.keras.models.load_model(model_path)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        y_pred_probs = model.predict(x_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
    except (ImportError, Exception) as e:
        print(f"TensorFlow not loaded ({e}). Evaluating model from HDF5 weights...")
        import h5py

        with h5py.File(model_path, "r") as hf:
            g = hf["model_weights"]
            w1 = g["w1"][:]
            b1 = g["b1"][:]
            w2 = g["w2"][:]
            b2 = g["b2"][:]

        x_te_flat = x_test.reshape(-1, 784).astype(np.float32)
        # Forward inference (no dropout during test evaluation)
        z1 = np.dot(x_te_flat, w1) + b1
        a1 = np.maximum(0, z1)
        z2 = np.dot(a1, w2) + b2
        exp_z = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
        probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        n_samples = len(y_test)
        loss = -np.mean(np.log(np.clip(probs[np.arange(n_samples), y_test], 1e-12, 1.0)))
        y_pred = np.argmax(probs, axis=1)
        accuracy = np.mean(y_pred == y_test)

    print(f"Test Loss: {loss:.4f} | Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    cm = compute_confusion_matrix(y_test, y_pred)

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
