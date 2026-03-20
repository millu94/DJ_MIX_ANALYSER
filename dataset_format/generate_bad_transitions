import os
import random
from pydub import AudioSegment
from pydub.effects import normalize

def generate_bad_dataset(input_folder, output_folder):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all WAV files in the 'Good' folder
    good_files = [f for f in os.listdir(input_folder) if f.endswith('.wav')]
    
    print(f"Found {len(good_files)} good segments. Generating bad segments...")

    for i, file_name in enumerate(good_files):
        # 1. Load the 'Base' segment
        base_path = os.path.join(input_folder, file_name)
        seg1 = AudioSegment.from_wav(base_path)

        # 2. Pick a random 'Overlay' segment (different from the base)
        random_file = random.choice(good_files)
        while random_file == file_name:
            random_file = random.choice(good_files)
            
        overlay_path = os.path.join(input_folder, random_file)
        seg2 = AudioSegment.from_wav(overlay_path)

        # 3. Reduce volume by 3dB-6dB each to prevent digital clipping when summed
        seg1 = seg1 - 4 
        seg2 = seg2 - 4

        # 4. Overlay them 
        # We add a tiny random offset (0-500ms) to ensure they aren't perfectly aligned
        bad_segment = seg1.overlay(seg2, position=random.randint(0, 500))

        # 5. Normalize to -1.0 dB
        # This is CRITICAL so the model focuses on rhythm, not volume
        final_segment = normalize(bad_segment)

        # 6. Export
        output_name = f"bad_transition_{i}.wav"
        final_segment.export(
            os.path.join(output_folder, output_name),
            format="wav",
            parameters=["-ar", "44100", "-ac", "1", "-sample_fmt", "s16"]
        )

    print(f"Done! Created {len(good_files)} bad segments in {output_folder}")

# Usage
generate_bad_dataset("Main Phase - RA Mix, Good Chunks", "Main Phase - RA Mix, Bad Chunks")