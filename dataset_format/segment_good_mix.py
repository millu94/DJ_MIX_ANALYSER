import os
from pydub import AudioSegment
from pydub.effects import normalize

def chop_mix(file_path, mix_name, output_dir):
    print(f"Loading {mix_name}...")
    mix = AudioSegment.from_mp3(file_path)
    
    # resample to mono to reduce data size and focus on rhythmic features
    mix = mix.set_channels(1)
    
    # the mvp requires 30s, 60s, 90s, and 120s segments
    segment_lengths = [30, 60, 90, 120]

    for seg_sec in segment_lengths:
        seg_ms = seg_sec * 1000
        
        for i, start_ms in enumerate(range(0, len(mix), seg_ms)):
            end_ms = start_ms + seg_ms
            
            # skip the final chunk if it doesn't meet the full length
            if end_ms > len(mix):
                break
                
            chunk = mix[start_ms:end_ms]
            normalized_chunk = normalize(chunk)
            
            # apply a 5ms fade in/out to prevent the model from detecting edit points
            faded_chunk = normalized_chunk.fade_in(5).fade_out(5)
            
            # format: {label}_{mix_name}_{seg_length}_{number}.flac (label 1 = good)
            output_name = f"1_{mix_name}_{seg_sec}_{i}.flac"
            
            # export as 16-bit mono flac at 22.05khz
            faded_chunk.export(
                os.path.join(output_dir, output_name),
                format="flac",
                parameters=["-ar", "22050", "-ac", "1", "-sample_fmt", "s16"]
            )
            
    print(f"Finished segmenting {mix_name} into 30/60/90/120s chunks.")