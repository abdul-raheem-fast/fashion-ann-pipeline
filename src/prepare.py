import os
import numpy as np
import tensorflow as tf

def prepare():
    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    print("Downloading/loading Fashion-MNIST dataset from tf.keras.datasets...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    print(f"Raw train shape: {x_train.shape}, Raw test shape: {x_test.shape}")

    np.save(os.path.join(raw_dir, "x_train.npy"), x_train)
    np.save(os.path.join(raw_dir, "y_train.npy"), y_train)
    np.save(os.path.join(raw_dir, "x_test.npy"), x_test)
    np.save(os.path.join(raw_dir, "y_test.npy"), y_test)

    print(f"Successfully saved raw arrays to '{raw_dir}'.")

if __name__ == "__main__":
    prepare()
