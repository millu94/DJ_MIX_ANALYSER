import os
import random
import multiprocessing
from pydub import AudioSegment
from pydub.effects import normalize

def process_one_segment(args):
    try:
        """
        Processes a single 'Bad' segment by either clashing two segments 
        from the same mix or inserting a gap.
        """
        idx, input_folder, output_folder, mix_name, seg_length, good_files_in_group = args
        
        # 80% Clash within the same mix, 20% Silence Gap
        # error_type = random.choices(['clash', 'gap'], weights=[0.8, 0.2], k=1)[0]
        
        # if error_type == 'clash':

        # Pick two different segments from the SAME mix and SAME length
        file1, file2 = random.sample(good_files_in_group, 2)
        
        # Load the flac files
        seg1 = AudioSegment.from_file(os.path.join(input_folder, file1), format="flac")
        seg2 = AudioSegment.from_file(os.path.join(input_folder, file2), format="flac")

        # 1. Overlay with offset
        # We still drop them slightly before overlaying to prevent extreme clipping
        # before the master normalization happens.
        combined = (seg1 - 3).overlay((seg2 - 3), position=random.randint(100, 1000))

        """
        Once model is 90% accurate at detecting Beat Clashing, move to a "Regressive" or 
        "Multi-Class" model. At that stage, you can re-introduce:
        Short Gaps (500ms): "Technical slip-ups."
        Long Gaps (4000ms): "Total failure/Dead air."
        Phasing (100ms offset): "Poor beat-matching."
        """

        # 2. Re-Normalize the final result to 0dB
        # This ensures Label 0 and Label 1 have the exact same peak volume
        # Apply a 5ms fade in/out to match the 'Good' segments
        bad_segment = normalize(combined).fade_in(5).fade_out(5)

        # Use the idx passed from the task queue to ensure a unique filename
        output_name = f"0_{mix_name}_{seg_length}_{idx}.flac"
        bad_segment.export(os.path.join(output_folder, output_name), format="flac")
        return True
    except Exception as e:
        # Using print(e) to see exactly what went wrong in the worker
        print(f"Error processing {args[3]}: {e}")
        return False

def generate_bad_dataset(input_folder, output_folder):
    """
    Generates 'Bad' segments to perfectly match the count of 'Good' segments
    per mix and per segment length.
    """
    if not os.path.exists(input_folder):
        print(f"Error: Path {input_folder} does not exist.")
        return

    all_good_files = [f for f in os.listdir(input_folder) if f.startswith('1_') and f.endswith('.flac')]
    
    # Group good files by (mix_name, seg_length)
    groups = {}
    for f in all_good_files:
        parts = f.split('_')
        if len(parts) >= 3:
            key = (parts[1], parts[2]) # (mix_name, seg_length)
            if key not in groups:
                groups[key] = []
            groups[key].append(f)

    # Collect ALL tasks into one list first
    all_tasks = []
    for (mix_name, seg_length), good_files in groups.items():
        # Check how many bad files already exist for THIS specific group
        # We look for the pattern: 0_MixName_Length_
        prefix = f"0_{mix_name}_{seg_length}_"
        existing_bad = [f for f in os.listdir(output_folder) if f.startswith(prefix)]
        
        num_needed = len(good_files) - len(existing_bad)
        
        if num_needed > 0 and len(good_files) >= 2:
            print(f"Queuing {num_needed} bad segments for {mix_name} ({seg_length}s)")
            # We start numbering from the count of existing files to avoid overwriting
            for i in range(len(existing_bad), len(good_files)):
                all_tasks.append((i, input_folder, output_folder, mix_name, seg_length, good_files))

    if not all_tasks:
        print("Dataset is already balanced.")
        return

    print(f"🚀 Starting parallel generation of {len(all_tasks)} segments...")

    # For macOS/Python 3.12, we MUST use 'spawn' and use the 'with' context manager
    # This prevents the "RuntimeError: An attempt has been made to start a new process..."
    ctx = multiprocessing.get_context("spawn")
    num_cores = multiprocessing.cpu_count()
    
    with ctx.Pool(processes=num_cores) as pool:
        # imap_unordered is more memory efficient for large datasets
        for _ in pool.imap_unordered(process_one_segment, all_tasks):
            pass
    
    print("✅ All bad segments generated.")

# Standard Python idiom to prevent multiprocessing recursion
if __name__ == "__main__":
    # Drive paths
    TARGET_DIR = "/Volumes/Bitch/datasets/djmix_dataset"
    generate_bad_dataset(TARGET_DIR, TARGET_DIR)