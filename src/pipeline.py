import os
from typing import Tuple

import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from preprocessing import Preprocessing


def extract_features(
    audio_dir: str,
    labels_csv: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load audio files and compute a simple feature vector for each file.

    The labels CSV is expected to have at least two columns:
    - 'filename': name of the audio file (e.g. 'example.wav')
    - 'label': target label for that file
    """
    df = pd.read_csv(labels_csv)

    if "filename" not in df.columns or "label" not in df.columns:
        raise ValueError("labels_csv must contain 'filename' and 'label' columns.")

    feature_rows = []
    y_labels = []

    for _, row in df.iterrows():
        file_path = os.path.join(audio_dir, row["filename"])
        if not os.path.exists(file_path):
            # Skip missing files but keep going
            continue

        signal, sr = librosa.load(file_path, sr=None)

        zcr = Preprocessing.zero_crossing_rate(signal)
        centroid = Preprocessing.spectral_centroid(signal, sr)
        bandwidth = Preprocessing.spectral_bandwidth(signal, sr)

        feature_rows.append([zcr, centroid, bandwidth])
        y_labels.append(row["label"])

    if not feature_rows:
        raise RuntimeError("No features could be extracted. Check your audio_dir and labels_csv.")

    X = np.asarray(feature_rows, dtype=float)
    y = np.asarray(y_labels)
    return X, y


def build_datasets(
    audio_dir: str,
    labels_csv: str,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Create train/val/test splits with standardized features.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test, scaler
    """
    X, y = extract_features(audio_dir=audio_dir, labels_csv=labels_csv)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # First split off test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_scaled,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Then split train/val from remaining data
    val_relative = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_relative,
        random_state=random_state,
        stratify=y_train_val,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


if __name__ == "__main__":
    """
    This script will:
    - read audio files and labels
    - extract simple features (zero-crossing rate, spectral centroid, bandwidth)
    - standardize the features
    - split into train/validation/test sets
    """
    # Adjust these paths to match your local data layout.
    AUDIO_DIR = "data/audio"
    LABELS_CSV = "data/labels.csv"

    X_train, X_val, X_test, y_train, y_val, y_test, scaler = build_datasets(
        audio_dir=AUDIO_DIR,
        labels_csv=LABELS_CSV,
    )

    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)
    print("X_test shape:", X_test.shape)
    print("Number of classes in training set:", len(set(y_train)))