import os

# Define your directory
RAW_DIR = "/Volumes/Bitch/datasets/djmix_dataset"

def tidy_filenames(directory):
    count = 0
    # List all files in the directory
    files = os.listdir(directory)
    
    print(f"Starting cleanup in {directory}...")

    for filename in files:
        # Check if the file contains the suffixes you want to remove
        if "Clash" in filename or "Gap" in filename:
            # Create the new name by replacing the strings with an empty string
            new_name = filename.replace("Clash", "").replace("Gap", "")
            
            # Full paths for renaming
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)
            
            try:
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} -> {new_name}")
                count += 1
            except Exception as e:
                print(f"Error renaming {filename}: {e}")

    print(f"Cleanup complete. Total files renamed: {count}")

if __name__ == "__main__":
    tidy_filenames(RAW_DIR)