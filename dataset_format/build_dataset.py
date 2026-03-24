import os
from pydub import AudioSegment
from segment_good_mix import chop_mix
from generate_bad_transitions import generate_bad_dataset

# Harddrive paths
RAW_DIR = "/Volumes/Bitch/datasets/raw_djmix_dataset"
OUT_DIR = "/Volumes/Bitch/datasets/djmix_dataset"

def main():
    # Environment Check
    try:
        AudioSegment.silent(duration=10).export("test.wav", format="wav")
        os.remove("test.wav")
    except Exception as e:
        print(f"Error: {e}")
        return

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    # Check for existing "Good" segments (Label 1)
    existing_good_files = [f for f in os.listdir(OUT_DIR) if f.startswith('1_') and f.endswith('.flac')]
    
    # --- Phase 1: Segmenting Good Mixes ---
    if len(existing_good_files) > 0:
        print(f"\n--- Phase 1: SKIPPED ---")
        print(f"Found {len(existing_good_files)} existing good segments. Moving to Phase 2.")
    else:
        print("\n--- Phase 1: Segmenting Good Mixes ---")
        if os.path.exists(RAW_DIR):
            mp3_files = [f for f in os.listdir(RAW_DIR) if f.endswith('.mp3') and not f.startswith('.')]
            for file_name in mp3_files:
                mix_name = file_name.replace('.mp3', '').replace(' ', '')
                file_path = os.path.join(RAW_DIR, file_name)
                print(f"Processing: {file_name}")
                chop_mix(file_path, mix_name, OUT_DIR)
            
            # Update the count after processing
            existing_good_files = [f for f in os.listdir(OUT_DIR) if f.startswith('1_')]
        else:
            print(f"Could not find raw directory at {RAW_DIR}")
            return

    # --- Phase 2: Generating Bad Transitions ---
    print("\n--- Phase 2: Generating Bad Transitions ---")

    num_needed = len(existing_good_files)  

    if num_needed > 0:
        generate_bad_dataset(OUT_DIR, OUT_DIR, num_to_generate=num_needed)
    else:
        print("Dataset is already balanced or bad segments already exist. No new files generated.")

if __name__ == "__main__":
    main()