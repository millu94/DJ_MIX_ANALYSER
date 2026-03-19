from pydub import AudioSegment

try:
    # Try to load a tiny slice of your MP3
    # (Replace 'your_mix.mp3' with an actual filename)
    audio = AudioSegment.from_mp3("Main Phase - RA Mix.mp3")
    print("Success! Pydub and FFmpeg are working together.")
except Exception as e:
    print(f"Error: {e}")
    print("Hint: You likely need to install FFmpeg or add it to your PATH.")