# datasets.py --------------------------------------------------------------
from pathlib import Path
import csv
import urllib.request
import numpy as np


DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "uci"
DATA_DIR.mkdir(parents=True, exist_ok=True)

WDBC_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "breast-cancer-wisconsin/wdbc.data"
)
IONO_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data"


def _download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)


def _load_wdbc_raw(path):
    xs = []
    ys = []
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 32:
                continue
            label = 1.0 if row[1].strip().upper() == "M" else 0.0
            feats = [float(x) for x in row[2:32]]
            xs.append(feats)
            ys.append(label)
    if not xs:
        raise ValueError(f"No data loaded from {path}")
    return np.asarray(xs), np.asarray(ys)


def load_breast_cancer(
    test_frac=0.2,
    seed=0,
    standardize=True,
    add_intercept=True,
    dtype=np.float32,
):
    """
    Load the UCI Wisconsin Diagnostic Breast Cancer dataset (WDBC).

    Returns dict with X_train, y_train, X_test, y_test.
    """
    data_path = DATA_DIR / "wdbc.data"
    if not data_path.exists():
        _download(WDBC_URL, data_path)

    X, y = _load_wdbc_raw(data_path)

    rng = np.random.default_rng(seed)
    n = X.shape[0]
    n_test = max(1, int(n * test_frac))
    perm = rng.permutation(n)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    X_train = X[train_idx].astype(dtype)
    y_train = y[train_idx].astype(dtype)
    X_test = X[test_idx].astype(dtype)
    y_test = y[test_idx].astype(dtype)

    if standardize:
        mean = X_train.mean(axis=0, keepdims=True)
        std = X_train.std(axis=0, keepdims=True)
        std = np.where(std == 0, 1.0, std)
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

    if add_intercept:
        ones_train = np.ones((X_train.shape[0], 1), dtype=dtype)
        ones_test = np.ones((X_test.shape[0], 1), dtype=dtype)
        X_train = np.concatenate([ones_train, X_train], axis=1)
        X_test = np.concatenate([ones_test, X_test], axis=1)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }


def _load_ionosphere_raw(path):
    xs = []
    ys = []
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 35:
                continue
            feats = [float(x) for x in row[:34]]
            label = row[34].strip().lower()
            ys.append(1.0 if label == "g" else 0.0)
            xs.append(feats)
    if not xs:
        raise ValueError(f"No data loaded from {path}")
    return np.asarray(xs), np.asarray(ys)


def load_ionosphere(
    test_frac=0.2,
    seed=0,
    standardize=True,
    add_intercept=True,
    dtype=np.float32,
):
    """
    Load the UCI Ionosphere dataset.

    Returns dict with X_train, y_train, X_test, y_test.
    """
    data_path = DATA_DIR / "ionosphere.data"
    if not data_path.exists():
        _download(IONO_URL, data_path)

    X, y = _load_ionosphere_raw(data_path)

    rng = np.random.default_rng(seed)
    n = X.shape[0]
    n_test = max(1, int(n * test_frac))
    perm = rng.permutation(n)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    X_train = X[train_idx].astype(dtype)
    y_train = y[train_idx].astype(dtype)
    X_test = X[test_idx].astype(dtype)
    y_test = y[test_idx].astype(dtype)

    if standardize:
        mean = X_train.mean(axis=0, keepdims=True)
        std = X_train.std(axis=0, keepdims=True)
        std = np.where(std == 0, 1.0, std)
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

    if add_intercept:
        ones_train = np.ones((X_train.shape[0], 1), dtype=dtype)
        ones_test = np.ones((X_test.shape[0], 1), dtype=dtype)
        X_train = np.concatenate([ones_train, X_train], axis=1)
        X_test = np.concatenate([ones_test, X_test], axis=1)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }
