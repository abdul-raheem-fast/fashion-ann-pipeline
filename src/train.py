import os
import csv
import yaml
import numpy as np

def load_train_params(params_path="params.yaml"):
    with open(params_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    return params.get("train", {})

def train():
    params = load_train_params()
    dense_units = int(params.get("dense_units", 128))
    dropout_rate = float(params.get("dropout_rate", 0.2))
    learning_rate = float(params.get("learning_rate", 0.001))
    epochs = int(params.get("epochs", 10))
    batch_size = int(params.get("batch_size", 64))

    processed_dir = os.path.join("data", "processed")
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    print("Loading processed dataset splits...")
    x_train = np.load(os.path.join(processed_dir, "x_train.npy"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    x_val = np.load(os.path.join(processed_dir, "x_val.npy"))
    y_val = np.load(os.path.join(processed_dir, "y_val.npy"))

    print(f"Hyperparameters: units={dense_units}, dropout={dropout_rate}, lr={learning_rate}, epochs={epochs}, batch={batch_size}")

    model_path = os.path.join(models_dir, "model.h5")
    history_path = os.path.join(models_dir, "history.csv")

    try:
        import tensorflow as tf
        print("Constructing Sequential ANN architecture with TensorFlow/Keras...")
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(dense_units, activation="relu"),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(10, activation="softmax")
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        model.summary()

        print("Training model with Keras...")
        history = model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_val, y_val),
            verbose=1
        )

        print(f"Saving trained model weights to '{model_path}'...")
        model.save(model_path)

        keys = list(history.history.keys())
        with open(history_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(keys)
            for i in range(len(history.history[keys[0]])):
                writer.writerow([history.history[k][i] for k in keys])

    except (ImportError, Exception) as e:
        print(f"TensorFlow runtime not detected ({e}). Training Sequential ANN via NumPy Adam engine...")
        import h5py

        # Flatten inputs: (N, 784)
        x_tr_flat = x_train.reshape(-1, 784).astype(np.float32)
        x_va_flat = x_val.reshape(-1, 784).astype(np.float32)

        np.random.seed(42)
        # He initialization for ReLU layer, Xavier for Softmax
        w1 = np.random.randn(784, dense_units).astype(np.float32) * np.sqrt(2.0 / 784.0)
        b1 = np.zeros((dense_units,), dtype=np.float32)
        w2 = np.random.randn(dense_units, 10).astype(np.float32) * np.sqrt(2.0 / (dense_units + 10))
        b2 = np.zeros((10,), dtype=np.float32)

        # Adam optimizer state
        mw1, vw1 = np.zeros_like(w1), np.zeros_like(w1)
        mb1, vb1 = np.zeros_like(b1), np.zeros_like(b1)
        mw2, vw2 = np.zeros_like(w2), np.zeros_like(w2)
        mb2, vb2 = np.zeros_like(b2), np.zeros_like(b2)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t = 0

        history_rows = []
        n_samples = len(x_tr_flat)

        for epoch in range(1, epochs + 1):
            perm = np.random.permutation(n_samples)
            total_loss = 0.0
            correct = 0

            for i in range(0, n_samples, batch_size):
                idx = perm[i:i + batch_size]
                xb, yb = x_tr_flat[idx], y_train[idx]
                bsize = len(xb)

                # Forward: Flatten -> Dense 1 -> ReLU -> Dropout
                z1 = np.dot(xb, w1) + b1
                a1 = np.maximum(0, z1)
                mask = (np.random.rand(*a1.shape) >= dropout_rate).astype(np.float32) / (1.0 - dropout_rate)
                a1_drop = a1 * mask

                # Dense 2 -> Softmax
                z2 = np.dot(a1_drop, w2) + b2
                exp_z = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
                probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)

                # Loss & Accuracy
                loss = -np.mean(np.log(np.clip(probs[np.arange(bsize), yb], 1e-12, 1.0)))
                total_loss += loss * bsize
                preds = np.argmax(probs, axis=1)
                correct += np.sum(preds == yb)

                # Backward pass
                dz2 = probs.copy()
                dz2[np.arange(bsize), yb] -= 1.0
                dz2 /= bsize

                dw2 = np.dot(a1_drop.T, dz2)
                db2 = np.sum(dz2, axis=0)

                da1 = np.dot(dz2, w2.T) * mask
                dz1 = da1 * (z1 > 0)
                dw1 = np.dot(xb.T, dz1)
                db1 = np.sum(dz1, axis=0)

                # Adam updates
                t += 1
                for param, grad, m, v in [(w1, dw1, mw1, vw1), (b1, db1, mb1, vb1),
                                          (w2, dw2, mw2, vw2), (b2, db2, mb2, vb2)]:
                    m[...] = beta1 * m + (1.0 - beta1) * grad
                    v[...] = beta2 * v + (1.0 - beta2) * (grad ** 2)
                    m_hat = m / (1.0 - beta1 ** t)
                    v_hat = v / (1.0 - beta2 ** t)
                    param -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)

            train_loss = total_loss / n_samples
            train_acc = correct / n_samples

            # Validation evaluation
            z1_v = np.dot(x_va_flat, w1) + b1
            a1_v = np.maximum(0, z1_v)
            z2_v = np.dot(a1_v, w2) + b2
            exp_zv = np.exp(z2_v - np.max(z2_v, axis=1, keepdims=True))
            probs_v = exp_zv / np.sum(exp_zv, axis=1, keepdims=True)
            val_loss = -np.mean(np.log(np.clip(probs_v[np.arange(len(y_val)), y_val], 1e-12, 1.0)))
            val_acc = np.mean(np.argmax(probs_v, axis=1) == y_val)

            print(f"Epoch {epoch}/{epochs} - loss: {train_loss:.4f} - accuracy: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}")
            history_rows.append({
                "loss": train_loss,
                "accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            })

        print(f"Saving trained model weights to '{model_path}'...")
        with h5py.File(model_path, "w") as hf:
            hf.attrs["architecture"] = "Flatten -> Dense(ReLU) -> Dropout -> Dense(10, Softmax)"
            hf.attrs["dense_units"] = dense_units
            hf.attrs["dropout_rate"] = dropout_rate
            g = hf.create_group("model_weights")
            g.create_dataset("w1", data=w1)
            g.create_dataset("b1", data=b1)
            g.create_dataset("w2", data=w2)
            g.create_dataset("b2", data=b2)

        print(f"Saving training history to '{history_path}'...")
        with open(history_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["loss", "accuracy", "val_loss", "val_accuracy"])
            writer.writeheader()
            writer.writerows(history_rows)

    print("Training phase completed successfully.")

if __name__ == "__main__":
    train()
