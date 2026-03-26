import os
import shutil
import random
from collections import defaultdict

# Harddrive paths
RAW_DIR = "/Volumes/Bitch/datasets/djmix_dataset"
OUT_DIR = "/Volumes/Bitch/datasets/djmix_dataset_partition"

def get_metadata(filename):
    """Extracts Label, Title, and Length from filename."""
    parts = filename.split('_')
    if len(parts) >= 3:
        label = parts[0]
        title = parts[1]
        seg_length = parts[2]
        return label, title, seg_length
    return None, None, None

def main():
    if not os.path.exists(RAW_DIR):
        print(f"Error: Drive not found at {RAW_DIR}")
        return

    # 1. Prepare Clean Output Directory
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR)

    all_files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith('.flac')]
    
    # 2. Group files by (Title, Length)
    groups = defaultdict(lambda: defaultdict(list))
    for f in all_files:
        label, title, seg_len = get_metadata(f)
        if label in ["0", "1"] and title and seg_len:
            groups[(title, seg_len)][label].append(f)

    selected_files = []

    # 3. Print Header
    print("="*85)
    print(f"{'MIX TITLE':<25} | {'LEN':<5} | {'TOTAL (G/B)':<15} | {'10% SAMPLE (1:1)'}")
    print("-" * 85)

    # 4. Process every mix/length combo
    for (title, seg_len), labels in sorted(groups.items()):
        list_0 = labels.get('0', [])
        list_1 = labels.get('1', [])
        
        # We need at least one of each to make a pair
        min_available = min(len(list_0), len(list_1))
        
        if min_available == 0:
            continue

        # CALCULATE 10% FOR THIS SPECIFIC MIX/LENGTH
        # max(1, ...) ensures that even if a mix is small, it still contributes at least 1 pair
        sample_size = max(1, int(min_available * 0.10))

        batch_0 = random.sample(list_0, sample_size)
        batch_1 = random.sample(list_1, sample_size)
        
        selected_files.extend(batch_0 + batch_1)
        
        counts_str = f"{len(list_1)}G / {len(list_0)}B"
        print(f"{title[:25]:<25} | {seg_len:<5} | {counts_str:<15} | {sample_size} pairs")

    # 5. Execute Copy
    print("-" * 85)
    print(f"Total files in balanced 10% partition: {len(selected_files)}")
    print(f"Copying to {OUT_DIR}...")
    
    for f in selected_files:
        shutil.copy2(os.path.join(RAW_DIR, f), os.path.join(OUT_DIR, f))

    print("\n✅ Success: 10% even-split partition complete.")

if __name__ == "__main__":
    main()