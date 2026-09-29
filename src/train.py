import os
import csv
import yaml
import numpy as np
import tensorflow as tf

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

    print("Constructing Sequential ANN architecture...")
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

    print("Training model...")
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        verbose=1
    )

    model_path = os.path.join(models_dir, "model.h5")
    print(f"Saving trained model weights to '{model_path}'...")
    model.save(model_path)

    history_path = os.path.join(models_dir, "history.csv")
    print(f"Saving training history to '{history_path}'...")
    keys = list(history.history.keys())
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for i in range(len(history.history[keys[0]])):
            writer.writerow([history.history[k][i] for k in keys])

    print("Training phase completed successfully.")

if __name__ == "__main__":
    train()
