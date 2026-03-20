from pydub import AudioSegment
import os
from pydub.effects import normalize


def chop_mix(file_path, segment_length_sec, output_dir):
    # Load the MP3
    mix = AudioSegment.from_mp3(file_path)
    
    # Segment length in milliseconds
    seg_ms = segment_length_sec * 1000
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, start_ms in enumerate(range(0, len(mix), seg_ms)):
        end_ms = start_ms + seg_ms
        chunk = mix[start_ms:end_ms]
        normalized_chunk = normalize(chunk)
        
        # Export as WAV (this handles the conversion)
        normalized_chunk.export(
            f"{output_dir}/good_transition_{i}.wav",
            format="wav",
            parameters=["-ar", "44100", "-ac", "1", "-sample_fmt", "s16"]
        )
        print(f"Exported chunk {i}")

# Usage: Chop a mix into 60-second segments
chop_mix("Main Phase - RA Mix.mp3", 90, "Main Phase - RA Mix, Good Transitions")