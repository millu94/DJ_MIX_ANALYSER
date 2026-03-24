import os
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from preprocessing import Preprocessing

FEATURE_COLUMNS = [
    "zcr",
    "spectral_centroid",
    "onset_env_mean",
    "onset_env_std",
    "tempogram_mean",
    "tempogram_std",
    "tempo_mean",
    "chroma_stft_mean",
    "chroma_stft_std",
    "chroma_cqt_mean",
    "chroma_cqt_std",
    "chroma_cens_mean",
    "chroma_cens_std",
    "mfcc_mean",
    "mfcc_std",
]


def collect_labeled_files(datasets_root: str):
    """
    Build a labeled file list from dataset folders.

    Expected folder names:
    - Main Phase - RA Mix, Good Transitions -> label 1
    - Main Phase - RA Mix, Bad Transitions -> label 0
    """
    root = Path(datasets_root)
    good_dir = root / "Main Phase - RA Mix, Good Transitions"
    bad_dir = root / "Main Phase - RA Mix, Bad Transitions"

    if not good_dir.exists() or not bad_dir.exists():
        raise RuntimeError(
            "Expected folders not found under datasets_root: "
            "'Main Phase - RA Mix, Good Transitions' and "
            "'Main Phase - RA Mix, Bad Transitions'."
        )

    labeled_files = []
    labeled_files.extend([(str(path), 1) for path in sorted(good_dir.glob("*.wav"))])
    labeled_files.extend([(str(path), 0) for path in sorted(bad_dir.glob("*.wav"))])

    if not labeled_files:
        raise RuntimeError("No .wav files found in the good/bad dataset folders.")

    return labeled_files


def extract_features(datasets_root: str):
    """
    Load audio files and compute simple feature vectors.
    Labels are inferred from folder:
    - good -> 1
    - bad -> 0
    """
    print("Extracting features...")
    feature_rows = []
    y_labels = []
    file_paths = []

    for file_path, label in collect_labeled_files(datasets_root):
        if not os.path.exists(file_path):
            continue
        
        print(f"Loading audio file: {file_path}")
        signal, sr = librosa.load(file_path, sr=None)

        zcr = float(Preprocessing.zero_crossing_rate(signal))
        centroid = float(Preprocessing.spectral_centroid(signal, sr))
        onset_env = np.asarray(Preprocessing.onset_strength_envelope(signal, sr), dtype=float)
        tempogram = np.asarray(Preprocessing.tempogram(signal, sr), dtype=float)
        tempo = np.asarray(Preprocessing.tempo(signal, sr), dtype=float)
        chroma_stft = np.asarray(Preprocessing.chroma_stft(signal, sr), dtype=float)
        chroma_cqt = np.asarray(Preprocessing.chroma_cqt(signal, sr), dtype=float)
        chroma_cens = np.asarray(Preprocessing.chroma_cens(signal, sr), dtype=float)
        mfcc = np.asarray(
            Preprocessing.mel_frequency_cepstral_coefficients(signal, sr),
            dtype=float,
        )

        # Summarize time/frequency matrices to fixed-size scalars per file.
        feature_vector = [
            zcr,
            centroid,
            float(np.mean(onset_env)),
            float(np.std(onset_env)),
            float(np.mean(tempogram)),
            float(np.std(tempogram)),
            float(np.mean(tempo)),
            float(np.mean(chroma_stft)),
            float(np.std(chroma_stft)),
            float(np.mean(chroma_cqt)),
            float(np.std(chroma_cqt)),
            float(np.mean(chroma_cens)),
            float(np.std(chroma_cens)),
            float(np.mean(mfcc)),
            float(np.std(mfcc)),
        ]
        feature_rows.append(feature_vector)
        y_labels.append(label)
        file_paths.append(file_path)

    if not feature_rows:
        raise RuntimeError(
            "No features could be extracted. Check dataset folder names and audio files."
        )

    X = np.asarray(feature_rows, dtype=float)
    y = np.asarray(y_labels, dtype=int)
    return X, y, file_paths


def save_processed_dataframe(X, y, file_paths, processed_dir: str = "Datasets/processed"):
    """
    Save extracted features and labels to a dataframe in Datasets/processed.
    """
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(X, columns=FEATURE_COLUMNS)
    df["label"] = y
    df["file_path"] = file_paths
    df["class_name"] = np.where(df["label"] == 1, "good", "bad")

    csv_path = processed_path / "features.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved processed dataframe to: {csv_path}")

    return df


def build_datasets(
    datasets_root: str,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
):
    """
    Create train/val/test splits with standardized features.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test, scaler
    """
    X, y, file_paths = extract_features(datasets_root=datasets_root)
    _ = save_processed_dataframe(X, y, file_paths, processed_dir="Datasets/processed")

    scaler = StandardScaler()
    print("Fitting scaler...")
    X_scaled = scaler.fit_transform(X)

    # First split off test set
    print("Splitting off test set...")
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

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    """
    This script will:
    - read audio files and labels
    - extract simple features (zero-crossing rate, spectral centroid, bandwidth)
    - standardize the features
    - split into train/validation/test sets
    """
    # Root folder that contains both good and bad transition folders.
    DATASETS_ROOT = "datasets"
    print("Building datasets...")
    X_train, X_val, X_test, y_train, y_val, y_test = build_datasets(
        datasets_root=DATASETS_ROOT
    )
    print("Datasets built successfully")
    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)
    print("X_test shape:", X_test.shape)
    print("Number of classes in training set:", len(set(y_train)))