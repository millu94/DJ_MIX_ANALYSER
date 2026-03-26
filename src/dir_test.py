import os
from pathlib import Path

def test_directory_logic():
    # 1. Get the current script's location
    # resolve() gets the absolute path
    script_dir = Path(__file__).resolve().parent
    
    # 2. Your .parent.parent logic
    # Level 0: dir_test.py
    # Level 1 (.parent): src/
    # Level 2 (.parent.parent): DJ_MIX_ANALYSER/ (The project root)
    project_root = script_dir.parent
    
    # 3. Define the target directory
    processed_dir = project_root / "datasets" / "processed"
    
    
    # 4. Attempt to create the directory
    try:
        os.makedirs(processed_dir, exist_ok=True)
        print(f"\n✅ Success: Directory created (or already exists).")
        
        # 5. Create a dummy file to verify in Finder
        test_file = processed_dir / "location_verification.txt"
        with open(test_file, "w") as f:
            f.write("If you see this, the path logic in pipeline.py is correct.")
        
        print(f"✅ Success: Verification file created at:\n   {test_file}")
        
    except Exception as e:
        print(f"\n❌ Error: Could not create directory. {e}")

if __name__ == "__main__":
    test_directory_logic()