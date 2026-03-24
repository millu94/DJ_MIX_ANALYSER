import os
import random
import multiprocessing
from pydub import AudioSegment
from pydub.effects import normalize

def process_one_segment(args):
    """
    Processes a single 'Bad' segment by either clashing two segments 
    from the same mix or inserting a gap.
    """
    i, input_folder, output_folder, mix_name, seg_length, good_files_for_this_mix = args
    
    # 80% Clash within the same mix, 20% Silence Gap
    error_type = random.choices(['clash', 'gap'], weights=[0.8, 0.2], k=1)[0]
    
    if error_type == 'clash':
        # Pick two different segments from the SAME mix and SAME length
        file1, file2 = random.sample(good_files_for_this_mix, 2)
        
        seg1 = AudioSegment.from_file(os.path.join(input_folder, file1), format="flac")
        seg2 = AudioSegment.from_file(os.path.join(input_folder, file2), format="flac")

        # Sum them (no timestretching, just raw overlay)
        # We use a random offset to ensure beats don't accidentally align
        bad_segment = (seg1 - 4).overlay((seg2 - 4), position=random.randint(100, 1000))
        output_name = f"0_{mix_name}Clash_{seg_length}_{i}.flac"

    else:
        # --- ERROR TYPE 2: THE ACCIDENTAL STOP (GAP) ---
        file1 = random.choice(good_files_for_this_mix)
        seg1 = AudioSegment.from_file(os.path.join(input_folder, file1), format="flac")
        
        gap_duration_ms = random.randint(1500, 4000)
        gap_start_ms = random.randint(2000, len(seg1) - gap_duration_ms - 2000)

        silence = AudioSegment.silent(duration=gap_duration_ms)
        bad_segment = seg1[:gap_start_ms] + silence + seg1[gap_start_ms + gap_duration_ms:]
        output_name = f"0_{mix_name}Gap_{seg_length}_{i}.flac"

    # Export with 5ms fades
    final_segment = normalize(bad_segment).fade_in(5).fade_out(5)
    final_segment.export(
        os.path.join(output_folder, output_name),
        format="flac",
        parameters=["-ar", "22050", "-ac", "1", "-sample_fmt", "s16"]
    )
    return True

def generate_bad_dataset(input_folder, output_folder, num_to_generate):
    all_good_files = [f for f in os.listdir(input_folder) if f.startswith('1_') and f.endswith('.flac')]
    
    # Group files by Mix Name and Segment Length to satisfy your constraint
    # Dict structure: {(mix_name, seg_length): [file1, file2...]}
    groups = {}
    for f in all_good_files:
        parts = f.split('_')
        if len(parts) >= 4:
            key = (parts[1], parts[2]) # (mix_name, seg_length)
            if key not in groups:
                groups[key] = []
            groups[key].append(f)

    # Filter out groups that don't have enough files to make a clash
    valid_keys = [k for k, v in groups.items() if len(v) >= 2]
    
    if not valid_keys:
        print("Error: Not enough segments within individual mixes to create clashes.")
        return

    print(f"🚀 Parallel generation: {num_to_generate} bad segments (Internal Mix Clashes & Gaps)")

    tasks = []
    for i in range(num_to_generate):
        # Pick a random mix and length group for this specific 'bad' sample
        mix_name, seg_length = random.choice(valid_keys)
        good_files_for_this_mix = groups[(mix_name, seg_length)]
        
        tasks.append((i, input_folder, output_folder, mix_name, seg_length, good_files_for_this_mix))

    # Run in parallel
    num_cores = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_cores) as pool:
        for _ in pool.imap_unordered(process_one_segment, tasks):
            pass

    print(f"✅ Done! Created {num_to_generate} balanced bad segments.")