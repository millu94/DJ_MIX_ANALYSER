import os
from pathlib import Path

def test_directory_logic():
    # get the current script's location
    # resolve() gets the absolute path
    script_dir = Path(__file__).resolve().parent
    
    # the .parent.parent logic
    # level 0: dir_test.py
    # level 1 (.parent): src/
    # level 2 (.parent.parent): dj_mix_analyser/ (the project root)
    project_root = script_dir.parent
    
    # define the target directory
    processed_dir = project_root / "datasets" / "processed"
    
    
    # attempt to create the directory
    try:
        os.makedirs(processed_dir, exist_ok=True)
        print(f"\n✅ Success: Directory created (or already exists).")
        
        # create a dummy file to verify in finder
        test_file = processed_dir / "location_verification.txt"
        with open(test_file, "w") as f:
            f.write("If you see this, the path logic in pipeline.py is correct.")
        
        print(f"✅ Success: Verification file created at:\n   {test_file}")
        
    except Exception as e:
        print(f"\n❌ Error: Could not create directory. {e}")

if __name__ == "__main__":
    test_directory_logic()