import os
import shutil
import random
from collections import defaultdict

# Harddrive paths
RAW_DIR = "/Volumes/Bitch/datasets/djmix_dataset_partition"

def get_metadata(filename):
    """
    Extracts info from: Label_Title_SegLength_Index
    Example: 0_RA.989Binh_60_2848 -> Label: 0, Title: RA.989Binh, SegLength: 60
    """
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

    all_files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(('.flac', '.wav', '.mp3'))]
    
    # Structure: stats[mix_title][seg_length][label] = count
    stats = defaultdict(lambda: defaultdict(lambda: {"0": 0, "1": 0}))

    for f in all_files:
        label, title, seg_len = get_metadata(f)
        if label in ["0", "1"] and title and seg_len:
            stats[title][seg_len][label] += 1

    # Print Summary Table
    print("="*75)
    print(f"{'MIX TITLE':<25} | {'LENGTH':<8} | {'BAD (0)':<8} | {'GOOD (1)':<8}")
    print("-" * 75)
    
    for title in sorted(stats.keys()):
        for length in sorted(stats[title].keys(), key=lambda x: int(x) if x.isdigit() else x):
            count_0 = stats[title][length]["0"]
            count_1 = stats[title][length]["1"]
            print(f"{title[:25]:<25} | {length + 's':<8} | {count_0:<8} | {count_1:<8}")
        print("-" * 75)

if __name__ == "__main__":
    main()