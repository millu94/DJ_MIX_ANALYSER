import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import librosa

# Assuming Preprocessing class is in a file named preprocessing.py
from preprocessing import Preprocessing

FEATURE_COLUMNS = [
    "zcr", "spectral_centroid", "onset_env_mean", "onset_env_std",
    "tempogram_mean", "tempogram_std", "tempo_mean", "chroma_stft_mean",
    "chroma_stft_std", "chroma_cqt_mean", "chroma_cqt_std", "chroma_cens_mean",
    "chroma_cens_std", "mfcc_mean", "mfcc_std",
]

def collect_labeled_files(datasets_root: str):
    """
    Parses files in datasets_root based on the format:
    label_mixname_seglength_num.flac (e.g., 1_MainPhase_10s_01.flac)
    """
    root = Path(datasets_root)
    if not root.exists():
        raise RuntimeError(f"Directory not found: {root}")

    labeled_files = []
    # Search for both .wav and .flac based on your new format
    audio_files = list(root.glob("*.flac")) + list(root.glob("*.wav"))

    for path in audio_files:
        filename = path.name
        try:
            # Splits "1_MainPhase_..." into ["1", "MainPhase", ...]
            label_part = filename.split('_')[0]
            label = int(label_part)
            labeled_files.append((str(path), label))
        except (ValueError, IndexError):
            print(f"Skipping file with invalid format: {filename}")
            continue

    if not labeled_files:
        raise RuntimeError(f"No valid labeled audio files found in {datasets_root}")

    print(f"Found {len(labeled_files)} files.")
    return labeled_files

def extract_features(datasets_root: str):
    print("Extracting features...")
    feature_rows = []
    y_labels = []
    file_paths = []

    files = collect_labeled_files(datasets_root)

    for file_path, label in files:
        print(f"Processing: {os.path.basename(file_path)} (Label: {label})")
        
        # sr=None preserves native sampling rate
        signal, sr = librosa.load(file_path, sr=None)

        # Feature Extraction using your Preprocessing utility
        zcr = float(Preprocessing.zero_crossing_rate(signal))
        centroid = float(Preprocessing.spectral_centroid(signal, sr))
        onset_env = np.asarray(Preprocessing.onset_strength_envelope(signal, sr), dtype=float)
        tempogram = np.asarray(Preprocessing.tempogram(signal, sr), dtype=float)
        tempo = np.asarray(Preprocessing.tempo(signal, sr), dtype=float)
        chroma_stft = np.asarray(Preprocessing.chroma_stft(signal, sr), dtype=float)
        chroma_cqt = np.asarray(Preprocessing.chroma_cqt(signal, sr), dtype=float)
        chroma_cens = np.asarray(Preprocessing.chroma_cens(signal, sr), dtype=float)
        mfcc = np.asarray(Preprocessing.mel_frequency_cepstral_coefficients(signal, sr), dtype=float)

        feature_vector = [
            zcr, centroid,
            float(np.mean(onset_env)), float(np.std(onset_env)),
            float(np.mean(tempogram)), float(np.std(tempogram)),
            float(np.mean(tempo)),
            float(np.mean(chroma_stft)), float(np.std(chroma_stft)),
            float(np.mean(chroma_cqt)), float(np.std(chroma_cqt)),
            float(np.mean(chroma_cens)), float(np.std(chroma_cens)),
            float(np.mean(mfcc)), float(np.std(mfcc)),
        ]
        
        feature_rows.append(feature_vector)
        y_labels.append(label)
        file_paths.append(file_path)

    X = np.asarray(feature_rows, dtype=float)
    y = np.asarray(y_labels, dtype=int)
    return X, y, file_paths

def save_processed_dataframe(X, y, file_paths, datasets_root, processed_dir="Datasets/processed"):
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(X, columns=FEATURE_COLUMNS)
    df["label"] = y
    df["file_path"] = file_paths
    df["class_name"] = np.where(df["label"] == 1, "good", "bad")

    # Clean up filename for the CSV
    root_name = Path(datasets_root).name
    csv_path = processed_path / f"{root_name}_features.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved features to: {csv_path}")
    return df

def build_datasets(datasets_root: str, test_size=0.2, val_size=0.2, random_state=42):
    X, y, file_paths = extract_features(datasets_root)
    _ = save_processed_dataframe(X, y, file_paths, datasets_root)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Stratified split to keep label distribution consistent
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )

    val_relative = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_relative, random_state=random_state, stratify=y_train_val
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def run_full_pipeline(datasets_root, test_size=0.2, val_size=0.1):
    print(f"🚀 Starting pipeline for: {datasets_root}")
    
    # 1. Extract & Save
    X, y, file_paths = extract_features(datasets_root)
    df = save_processed_dataframe(X, y, file_paths, datasets_root)

    # 2. Extract unique mix names for Grouped Splitting
    # e.g., '1_MainPhase_30s_01.flac' -> 'MainPhase'
    df['mix_group'] = df['file_path'].apply(lambda x: os.path.basename(x).split('_')[1])
    unique_mixes = df['mix_group'].unique()
    
    # 3. Triple Split on Mixes
    # First, separate the "Hold-out" Test Mixes
    train_val_mixes, test_mixes = train_test_split(
        unique_mixes, test_size=test_size, random_state=42
    )
    
    # Second, split the remaining mixes into Train and Val
    # We adjust the val_size relative to the remaining pool
    val_relative = val_size / (1.0 - test_size)
    train_mixes, val_mixes = train_test_split(
        train_val_mixes, test_size=val_relative, random_state=42
    )
    
    # 4. Map the mixes back to the actual data rows
    train_df = df[df['mix_group'].isin(train_mixes)]
    val_df   = df[df['mix_group'].isin(val_mixes)]
    test_df  = df[df['mix_group'].isin(test_mixes)]

    print("\n--- Leak-Proof Split Results ---")
    print(f"Mixes in Train: {len(train_mixes)} | Segments: {len(train_df)}")
    print(f"Mixes in Val:   {len(val_mixes)} | Segments: {len(val_df)}")
    print(f"Mixes in Test:  {len(test_mixes)} | Segments: {len(test_df)}")

    return train_df, val_df, test_df

if __name__ == "__main__":
    # CONFIGURATION AREA
    # Put things here that change based on who is running the script
    EXTERNAL_DRIVE_PATH = "/Volumes/Bitch/datasets/djmix_dataset_partition"

    if os.path.exists(EXTERNAL_DRIVE_PATH):
        train_data, test_data = run_full_pipeline(EXTERNAL_DRIVE_PATH)
        print(f"\n✅ Done! Train segments: {len(train_data)}")
    else:
        print(f"❌ Drive not found at {EXTERNAL_DRIVE_PATH}")