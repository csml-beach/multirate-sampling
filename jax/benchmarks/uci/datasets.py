# datasets.py --------------------------------------------------------------
from pathlib import Path
import csv
import urllib.request
import numpy as np
import shutil
import subprocess


DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "uci"
DATA_DIR.mkdir(parents=True, exist_ok=True)

WDBC_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "breast-cancer-wisconsin/wdbc.data"
)
IONO_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data"
SPAM_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
A5A_TRAIN_URL = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a5a"
A5A_TEST_URL = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a5a.t"


def _download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
        return
    except Exception as err:
        print(f"download warning: urllib failed ({err}); trying curl/wget")

    if shutil.which("curl"):
        result = subprocess.run(
            ["curl", "-L", "-o", str(tmp), url],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and tmp.exists():
            tmp.replace(dest)
            return
        print(f"download warning: curl failed ({result.returncode}) {result.stderr.strip()}")
    if shutil.which("wget"):
        result = subprocess.run(
            ["wget", "-O", str(tmp), url],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and tmp.exists():
            tmp.replace(dest)
            return
        print(f"download warning: wget failed ({result.returncode}) {result.stderr.strip()}")
    raise RuntimeError(f"Failed to download {url}")


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


def _load_spambase_raw(path):
    xs = []
    ys = []
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 58:
                continue
            feats = [float(x) for x in row[:57]]
            ys.append(float(row[57]))
            xs.append(feats)
    if not xs:
        raise ValueError(f"No data loaded from {path}")
    return np.asarray(xs), np.asarray(ys)


def _load_libsvm_dense(path):
    rows = []
    labels = []
    max_idx = 0
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            label = float(parts[0])
            feats = []
            for item in parts[1:]:
                idx_str, val_str = item.split(":")
                idx = int(idx_str)
                val = float(val_str)
                feats.append((idx, val))
                if idx > max_idx:
                    max_idx = idx
            rows.append(feats)
            labels.append(label)
    if not rows:
        raise ValueError(f"No data loaded from {path}")
    X = np.zeros((len(rows), max_idx), dtype=np.float32)
    for i, feats in enumerate(rows):
        for idx, val in feats:
            X[i, idx - 1] = val
    y = np.asarray(labels, dtype=np.float32)
    return X, y, max_idx


def load_spambase(
    test_frac=0.2,
    seed=0,
    standardize=True,
    add_intercept=True,
    dtype=np.float32,
):
    """
    Load the UCI Spambase dataset.

    Returns dict with X_train, y_train, X_test, y_test.
    """
    data_path = DATA_DIR / "spambase.data"
    if not data_path.exists():
        _download(SPAM_URL, data_path)

    X, y = _load_spambase_raw(data_path)

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


def load_a5a(
    seed=0,
    standardize=True,
    add_intercept=True,
    dtype=np.float32,
):
    """
    Load the LIBSVM a5a (Adult) dataset with the provided train/test split.

    Returns dict with X_train, y_train, X_test, y_test.
    """
    train_path = DATA_DIR / "a5a"
    test_path = DATA_DIR / "a5a.t"
    if not train_path.exists():
        _download(A5A_TRAIN_URL, train_path)
    if not test_path.exists():
        _download(A5A_TEST_URL, test_path)

    X_train, y_train, max_train = _load_libsvm_dense(train_path)
    X_test, y_test, max_test = _load_libsvm_dense(test_path)
    n_features = max(max_train, max_test)
    if X_train.shape[1] != n_features:
        pad = n_features - X_train.shape[1]
        X_train = np.pad(X_train, ((0, 0), (0, pad)), mode="constant")
    if X_test.shape[1] != n_features:
        pad = n_features - X_test.shape[1]
        X_test = np.pad(X_test, ((0, 0), (0, pad)), mode="constant")

    y_train = (y_train > 0).astype(dtype)
    y_test = (y_test > 0).astype(dtype)
    X_train = X_train.astype(dtype)
    X_test = X_test.astype(dtype)

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

    _ = seed  # unused; kept for API compatibility
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }
