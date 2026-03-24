import os
import random
import multiprocessing
from pydub import AudioSegment
from pydub.effects import normalize

def process_one_segment(args):
    """
    Processes a single 'Bad' segment by either clashing two segments 
    from the same mix or (inserting a gap, adv).
    """
    input_folder, output_folder, mix_name, seg_length, file1, file_list = args

    # 1. EXTRACT 'num' FROM file1 TO FIX NameError
    # e.g., '1_RA.907AIDA_30_5.flac' -> '5'
    num = file1.replace('.flac', '').split('_')[3]

    file2 = random.choice(file_list)
    # Ensure we don't accidentally mix a file with itself to create a "bad" transition
    while file2 == file1 and len(file_list) > 1:
        file2 = random.choice(file_list)
    
    seg1 = AudioSegment.from_file(os.path.join(input_folder, file1), format="flac")
    seg2 = AudioSegment.from_file(os.path.join(input_folder, file2), format="flac")

    # Sum them (no timestretching, just raw overlay)
    # We use a random offset to ensure beats don't accidentally align
    bad_segment = (seg1 - 4).overlay((seg2 - 4), position=random.randint(100, 1000))
    # 2. ENFORCE STRICT LENGTH (Fixes future tensor shape mismatch errors)
    # The overlay method can extend the audio duration. We slice it down to exactly match seg1
    bad_segment = bad_segment[:len(seg1)]
    output_name = f"0_{mix_name}_{seg_length}_{num}.flac"

    # Export with 5ms fades
    final_segment = normalize(bad_segment).fade_in(5).fade_out(5)
    final_segment.export(
        os.path.join(output_folder, output_name),
        format="flac",
        parameters=["-ar", "22050", "-ac", "1", "-sample_fmt", "s16"]
    )
    return True

def generate_bad_dataset(input_folder, output_folder, num_to_generate):
    """
    Generates 'bad' segments that match the count of 'good' segments
    per mix and per segment length
    """
    all_good_files = [f for f in os.listdir(input_folder) if f.startswith('1_') and f.endswith('.flac')]
    
    # Group files by Mix Name and Segment Length 
    groups = {}
    key_list = []
    for f in all_good_files:
        parts = f.split('_')
        if len(parts) >= 4:
            key = (parts[1], parts[2]) # (mix_name, seg_length)
            if key not in groups:
                groups[key] = []
                key_list.append(key)
            groups[key].append(f)


    tasks = []
    for (mix_name, seg_length), file_list in groups.items():
            # For every good file, we will create exactly one bad file task
        for file1 in file_list:
            tasks.append((input_folder, output_folder, mix_name, seg_length, file1, file_list))



    # Run in parallel
    num_cores = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_cores) as pool:
        for _ in pool.imap_unordered(process_one_segment, tasks):
            pass
