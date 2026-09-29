import os
import gzip
import urllib.request
import numpy as np

def load_idx_images(file_path):
    with gzip.open(file_path, "rb") as f:
        _ = f.read(16)
        buf = f.read()
        return np.frombuffer(buf, dtype=np.uint8).reshape(-1, 28, 28)

def load_idx_labels(file_path):
    with gzip.open(file_path, "rb") as f:
        _ = f.read(8)
        buf = f.read()
        return np.frombuffer(buf, dtype=np.uint8)

def prepare():
    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    print("Loading Fashion-MNIST dataset...")
    try:
        import tensorflow as tf
        print("Using tf.keras.datasets.fashion_mnist...")
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    except (ImportError, Exception) as e:
        print(f"TensorFlow not loaded ({e}). Using direct official Fashion-MNIST archive...")
        base_url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
        files = {
            "x_train": "train-images-idx3-ubyte.gz",
            "y_train": "train-labels-idx1-ubyte.gz",
            "x_test": "t10k-images-idx3-ubyte.gz",
            "y_test": "t10k-labels-idx1-ubyte.gz"
        }
        for key, fname in files.items():
            dest = os.path.join(raw_dir, fname)
            if not os.path.exists(dest):
                print(f"Fetching {fname}...")
                urllib.request.urlretrieve(base_url + fname, dest)

        x_train = load_idx_images(os.path.join(raw_dir, files["x_train"]))
        y_train = load_idx_labels(os.path.join(raw_dir, files["y_train"]))
        x_test = load_idx_images(os.path.join(raw_dir, files["x_test"]))
        y_test = load_idx_labels(os.path.join(raw_dir, files["y_test"]))

    print(f"Raw train shape: {x_train.shape}, Raw test shape: {x_test.shape}")

    np.save(os.path.join(raw_dir, "x_train.npy"), x_train)
    np.save(os.path.join(raw_dir, "y_train.npy"), y_train)
    np.save(os.path.join(raw_dir, "x_test.npy"), x_test)
    np.save(os.path.join(raw_dir, "y_test.npy"), y_test)

    # Clean up temporary gz archives if present
    for fname in ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]:
        gz_path = os.path.join(raw_dir, fname)
        if os.path.exists(gz_path):
            os.remove(gz_path)

    print(f"Successfully saved raw arrays to '{raw_dir}'.")

if __name__ == "__main__":
    prepare()
