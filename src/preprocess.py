import os
import yaml
import numpy as np

def load_params(params_path="params.yaml"):
    with open(params_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    return params.get("preprocess", {})

def perform_split(x, y, val_size=0.1, seed=42):
    try:
        from sklearn.model_selection import train_test_split
        return train_test_split(
            x, y, test_size=val_size, random_state=seed, stratify=y
        )
    except ImportError:
        print("scikit-learn not available, using deterministic numpy split...")
        np.random.seed(seed)
        n = len(x)
        indices = np.random.permutation(n)
        split_idx = int(n * (1 - val_size))
        train_idx, val_idx = indices[:split_idx], indices[split_idx:]
        return x[train_idx], x[val_idx], y[train_idx], y[val_idx]

def preprocess():
    params = load_params()
    val_size = float(params.get("val_size", 0.1))
    seed = int(params.get("seed", 42))

    raw_dir = os.path.join("data", "raw")
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    print("Loading raw numpy arrays...")
    x_train_raw = np.load(os.path.join(raw_dir, "x_train.npy"))
    y_train_raw = np.load(os.path.join(raw_dir, "y_train.npy"))
    x_test_raw = np.load(os.path.join(raw_dir, "x_test.npy"))
    y_test = np.load(os.path.join(raw_dir, "y_test.npy"))

    print(f"Normalizing pixel values to [0, 1] (raw min: {x_train_raw.min()}, max: {x_train_raw.max()})...")
    x_train_norm = x_train_raw.astype(np.float32) / 255.0
    x_test_norm = x_test_raw.astype(np.float32) / 255.0

    print(f"Splitting training data into train and validation sets (val_size={val_size}, seed={seed})...")
    x_train, x_val, y_train, y_val = perform_split(
        x_train_norm,
        y_train_raw,
        val_size=val_size,
        seed=seed
    )

    print(f"Processed splits -> Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test_norm.shape}")

    np.save(os.path.join(processed_dir, "x_train.npy"), x_train)
    np.save(os.path.join(processed_dir, "y_train.npy"), y_train)
    np.save(os.path.join(processed_dir, "x_val.npy"), x_val)
    np.save(os.path.join(processed_dir, "y_val.npy"), y_val)
    np.save(os.path.join(processed_dir, "x_test.npy"), x_test_norm)
    np.save(os.path.join(processed_dir, "y_test.npy"), y_test)

    print(f"Successfully saved processed data splits to '{processed_dir}'.")

if __name__ == "__main__":
    preprocess()
