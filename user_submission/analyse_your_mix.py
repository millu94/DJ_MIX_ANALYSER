import os
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np
from pydub import AudioSegment
from pydub.effects import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# --- path resolution ---
# get current directory (user_submission)
script_dir = Path(__file__).resolve().parent
# go up one level to the project root (dj_mix_analyser)
project_root = script_dir.parent
# add the 'src' folder to python's path so the process can import the pipeline
sys.path.append(str(project_root / "src"))

from pipeline import extract_features_parallel, save_processed_dataframe

def chop_user_mix(mp3_path, output_dir):
    """segments a fresh mix into continuous 30-second chunks."""
    print(f"\n🎧 Loading mix: {os.path.basename(mp3_path)}...")
    try:
        mix = AudioSegment.from_mp3(mp3_path).set_channels(1)
    except Exception as e:
        print(f"❌ Error loading MP3. Ensure you have ffmpeg installed. Details: {e}")
        sys.exit(1)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # clear out any old chunks from previous runs
    for f in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, f))
        
    seg_ms = 30 * 1000  # 30 second window
    step_ms = 15 * 1000 # move forward by only 15 seconds
    chunk_count = 0
        
    for start_ms in range(0, len(mix) - seg_ms, step_ms):
        end_ms = start_ms + seg_ms
        chunk = mix[start_ms:end_ms]
            
        chunk = mix[start_ms:end_ms]
        chunk = normalize(chunk).fade_in(5).fade_out(5)
        
        # the process use '9' as a dummy label so pipeline.py doesn't crash
        out_name = os.path.join(output_dir, f"9_UserMix_30_{chunk_count}.flac")
        chunk.export(out_name, format="flac", parameters=["-ar", "22050", "-sample_fmt", "s16"])
        chunk_count += 1
        
    print(f"✅ Created {chunk_count} continuous segments.")
    return chunk_count


def analyze_user_mix():
    print("="*60)
    print("🎛️  DJ MIX ANALYSER - USER SUBMISSION PORTAL 🎛️")
    print("="*60)
    
    mp3_path = input("\nEnter the full path to your MP3 file (drag and drop here):\n> ").strip()
    mp3_path = mp3_path.replace("'", "").replace('"', "")
    
    if not os.path.exists(mp3_path):
        print("❌ File not found. Please try again.")
        return

    # create temporary folders inside user_submission for the audio chunks
    temp_dir = script_dir / "temp_processing"
    user_audio_dir = temp_dir / "audio"
    
    # segmentation
    num_chunks = chop_user_mix(mp3_path, str(user_audio_dir))
    if num_chunks == 0:
        print("❌ Audio file was too short to analyze.")
        return
    
    # feature extraction (reuses the pipeline.py logic)
    print(f"\n🔬 Extracting audio features (This will take a moment)...")
    start_time = time.time()
    
    # suppress the massive printout from pipeline.py temporarily for a clean ux
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    X, y, file_paths = extract_features_parallel(str(user_audio_dir))
    user_df = save_processed_dataframe(X, y, file_paths, str(temp_dir))
    sys.stdout = original_stdout
    
    print(f"⏱️ Extraction completed in {(time.time() - start_time):.1f}s")
    
    # load the pre-trained model (the professional way)
    print(f"\n🧠 Loading pre-trained Machine Learning model...")
    model_path = project_root / "models" / "production_rf_model.joblib"
    scaler_path = project_root / "models" / "production_scaler.joblib"
    
    if not model_path.exists() or not scaler_path.exists():
        print("❌ Model files missing! Run modelsjosh.py first to generate them.")
        return
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("✅ Model loaded instantly.")
    
    # inference (prediction)
    print(f"\n🔍 Scanning mix for bad transitions...")
    drop_cols = ["label", "file_path", "class_name", "mix_group"]
    X_user = user_df.drop(columns=drop_cols, errors="ignore")
    X_user_scaled = scaler.transform(X_user)
    
    # get the raw probability of the segment being "bad" (class 0)
    probabilities_bad = model.predict_proba(X_user_scaled)[:, 0] 
    
    # ---  sensitivity dial  ---
    # default is 0.50 (50%). 
    # lower it to 0.35 to make the ai much more aggressive at catching trainwrecks!
    SENSITIVITY_THRESHOLD = 0.35 
    
    # output the report
    print("\n" + "="*60)
    print("📊 MIX ANALYSIS REPORT")
    print("="*60)
    
    issues_found = 0
    # the process sort the file paths so they are read in chronological order
    user_df = user_df.sort_values(by="file_path").reset_index(drop=True)
    
    for i, prob in enumerate(probabilities_bad):
        if prob >= SENSITIVITY_THRESHOLD: # trigger if confidence passes the custom dial
            issues_found += 1
            # calculate timestamps based on chunk index
            start_sec = i * 15
            end_sec = start_sec + 30
            start_str = f"{start_sec // 60:02d}:{start_sec % 60:02d}"
            end_str = f"{end_sec // 60:02d}:{end_sec % 60:02d}"
            confidence = prob * 100
            
            # change color/emoji based on severity
            if confidence > 65:
                print(f"🚨 CRITICAL Trainwreck: [{start_str} - {end_str}] (Confidence: {confidence:.1f}%)")
            else:
                print(f"⚠️ Minor sloppy mix: [{start_str} - {end_str}] (Confidence: {confidence:.1f}%)")
            
    if issues_found == 0:
        print(f"✨ Seamless! (At {SENSITIVITY_THRESHOLD*100}% sensitivity, no bad transitions detected).")
    else:
        print(f"\nTotal potential issues flagged: {issues_found}")
    print("="*60)

    # ---  google sheets export generator ---
    if issues_found > 0:
        print("\n📋 COPY/PASTE FOR GOOGLE SHEETS:")
        print("Mix Name,Start Time,Seconds,End Time,Type,Confidence (%), Comments")
        
        # re-run the loop logic to print clean csv rows
        for i, prob in enumerate(probabilities_bad):
            if prob >= SENSITIVITY_THRESHOLD:
                start_sec = i * 15
                end_sec = start_sec + 30
                start_str = f"{start_sec // 60:02d}:{start_sec % 60:02d}"
                end_str = f"{end_sec // 60:02d}:{end_sec % 60:02d}"
                
                label = "CRITICAL" if prob * 100 > 65 else "Minor"
                
                # using the original mp3 filename for the first column
                mix_name = os.path.basename(mp3_path)
                print(f"{mix_name},{start_str},{start_sec},{end_str},{label},{prob*100:.1f}%")

    # clear out temp directory
    for f in os.listdir(str(user_audio_dir)):
        os.remove(os.path.join(str(user_audio_dir), f))

if __name__ == "__main__":
    analyze_user_mix()