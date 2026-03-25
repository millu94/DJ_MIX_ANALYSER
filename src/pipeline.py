import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import librosa
import multiprocessing
from tqdm import tqdm  # You may need to: pip install tqdm

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

def process_one_file(file_info):
    """
    Worker function to process a single audio file.
    """
    file_path, label = file_info
    try:
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
        return feature_vector, label, file_path
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def extract_features_parallel(datasets_root: str):
    print("🚀 Starting Parallel Feature Extraction...")
    files = collect_labeled_files(datasets_root)
    
    # Use multiprocessing Pool
    num_cores = multiprocessing.cpu_count()
    results = []
    
    with multiprocessing.Pool(processes=num_cores) as pool:
        # imap_unordered is faster as it yields results as soon as they are ready
        for result in tqdm(pool.imap_unordered(process_one_file, files), total=len(files)):
            if result:
                results.append(result)

    # Unpack results
    feature_rows = [r[0] for r in results]
    y_labels = [r[1] for r in results]
    file_paths = [r[2] for r in results]

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

def run_manual_split_pipeline(datasets_root):
    # 1. Extract features from the external drive
    X, y, file_paths = extract_features_parallel(datasets_root) 
    df = save_processed_dataframe(X, y, file_paths, datasets_root)

    # 2. Add the mix name column for easy filtering
    # This assumes format: label_MixName_length_num.flac
    df['mix_group'] = df['file_path'].apply(lambda x: os.path.basename(x).split('_')[1])

    # 3. Define cohorts manually, this is to ensure there is't a bias over vinyl/digital
    test_mixes = ['RA.1002Nooriyah', 'RA.989Binh']
    val_mixes  = ['RA.1030MainPhase'] # Using one mix for tuning
    
    # Everything else goes to Training
    test_df  = df[df['mix_group'].isin(test_mixes)]
    val_df   = df[df['mix_group'].isin(val_mixes)]
    train_df = df[~df['mix_group'].isin(test_mixes + val_mixes)]

    print("\n--- Manual Split Summary ---")
    print(f"TRAIN: {train_df['mix_group'].unique()} ({len(train_df)} segments)")
    print(f"VAL:   {val_df['mix_group'].unique()} ({len(val_df)} segments)")
    print(f"TEST:  {test_df['mix_group'].unique()} ({len(test_df)} segments)")

    return train_df, val_df, test_df

if __name__ == "__main__":
    EXTERNAL_DRIVE_PATH = "/Volumes/Bitch/datasets/djmix_dataset_partition"

    if os.path.exists(EXTERNAL_DRIVE_PATH):
        # 1. Extract in parallel
        X, y, file_paths = extract_features_parallel(EXTERNAL_DRIVE_PATH)
        
        # 2. Save to CSV (for use in graphs.py and modelsdaniel.py)
        df = save_processed_dataframe(X, y, file_paths, EXTERNAL_DRIVE_PATH)
        
        # 3. Perform manual split for testing
        train_set, val_set, test_set = run_manual_split_pipeline(EXTERNAL_DRIVE_PATH)
        print(f"\n✅ Pipeline Complete. CSV generated at Datasets/processed/")
    else:
        print(f"❌ Drive not found at {EXTERNAL_DRIVE_PATH}")