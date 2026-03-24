import os
import shutil
import random
from collections import defaultdict

# Harddrive paths
RAW_DIR = "/Volumes/Bitch/datasets/djmix_dataset"
OUT_DIR = "/Volumes/Bitch/datasets/djmix_dataset_partition"

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

    # Print Table
    print("="*70)
    print(f"{'MIX TITLE':<25} | {'LENGTH':<8} | {'BAD (0)':<8} | {'GOOD (1)':<8}")
    print("-" * 70)
    
    # Sort by title first
    for title in sorted(stats.keys()):
        # For each mix, show the breakdown of lengths
        for length in sorted(stats[title].keys(), key=lambda x: int(x) if x.isdigit() else x):
            count_0 = stats[title][length]["0"]
            count_1 = stats[title][length]["1"]
            
            # Highlight imbalance within a specific mix
            diff = abs(count_0 - count_1)
            alert = "!" if diff > (max(count_0, count_1) * 0.5) else ""
            
            print(f"{title[:25]:<25} | {length + 's':<8} | {count_0:<8} | {count_1:<8} {alert}")
        print("-" * 70) # Separator between different mixes

    # if not os.path.exists(OUT_DIR):
    #     os.makedirs(OUT_DIR)

    # # 1. Group files by (SegLength, Label)
    # # We group by SegLength first to ensure we can balance 0s and 1s for every duration
    # all_files = [f for f in os.listdir(RAW_DIR) if f.endswith(('.flac', '.wav', '.mp3'))]
    
    # # Structure: master_dict[seg_length][label] = [list of files]
    # stats_map = defaultdict(lambda: defaultdict(list))
    
    # print(f"Found {len(all_files)} total audio files in {RAW_DIR}")
    # for f in all_files:
    #     label, title, seg_len = get_metadata(f)
    #     if label and seg_len:
    #         stats_map[seg_len][label].append(f)

    # # 2. Iterate through each segment length (e.g., "60", "120")
    
    # selected_files = []
    # # Dictionary to keep track of what we actually picked
    # final_stats = defaultdict(lambda: {"0": 0, "1": 0})

    # for seg_len, labels in stats_map.items():
    #     list_0 = labels.get('0', [])
    #     list_1 = labels.get('1', [])
        
    #     min_count = min(len(list_0), len(list_1))
    #     sample_size = max(1, int(min_count * 0.10))
        
    #     # Select the files
    #     batch_0 = random.sample(list_0, sample_size)
    #     batch_1 = random.sample(list_1, sample_size)
        
    #     selected_files.extend(batch_0)
    #     selected_files.extend(batch_1)

    #     # Update our stats counter
    #     final_stats[seg_len]["0"] = len(batch_0)
    #     final_stats[seg_len]["1"] = len(batch_1)

    # # --- PRINT THE SUMMARY TABLE ---
    # print("\n" + "="*40)
    # print(f"{'SEG LENGTH':<15} | {'BAD (0)':<10} | {'GOOD (1)':<10}")
    # print("-" * 40)
    
    # total_0 = 0
    # total_1 = 0
    
    # for length in sorted(final_stats.keys()):
    #     count_0 = final_stats[length]["0"]
    #     count_1 = final_stats[length]["1"]
    #     print(f"{length + 's':<15} | {count_0:<10} | {count_1:<10}")
    #     total_0 += count_0
    #     total_1 += count_1

    # print("-" * 40)
    # print(f"{'TOTAL':<15} | {total_0:<10} | {total_1:<10}")
    # print("="*40 + "\n")

    # # 3. Execution
    # print(f"\nCopying {len(selected_files)} balanced files to {OUT_DIR}...")
    # for f in selected_files:
    #     shutil.copy2(os.path.join(RAW_DIR, f), os.path.join(OUT_DIR, f))

    # print("Success: Dataset is partitioned and balanced by Segment Length.")

if __name__ == "__main__":
    main()