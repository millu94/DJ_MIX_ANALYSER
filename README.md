### AML Speakrightish

A small audio machine learning project for experimenting with basic preprocessing, dataset pipelines, and model training.

The current implementation focuses on:
- **Preprocessing**: simple, hand-crafted audio features (zero-crossing rate, spectral centroid, spectral bandwidth).
- **Pipeline**: turning a directory of audio files plus a labels CSV into standardized train/validation/test splits.
- **Models**: a placeholder for future training and evaluation code.

---

### Project structure

- `requirements.txt` – Python dependencies (NumPy, pandas, scikit-learn, librosa, torch, etc.).
- `src/preprocessing.py` – feature utilities (e.g. zero-crossing rate, spectral centroid, bandwidth).
- `src/pipeline.py` – minimal pipeline to load audio, extract features, standardize them, and create train/val/test splits.
- `src/models.py` – (planned) training and evaluation code for your models.

---

### Installation

1. **Create and activate a virtual environment** (recommended):

   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # on Windows
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

---

### Data layout

By default, the pipeline expects the following layout relative to the project root:

- `data/audio/` – directory containing your audio files (e.g. `example.wav`).
- `data/labels.csv` – a CSV file with at least:
  - `filename`: the name of the audio file (e.g. `example.wav`)
  - `label`: the target label for that file (string or integer)

Example `labels.csv`:

```csv
filename,label
sample_001.wav,0
sample_002.wav,1
```

You can change these paths by editing `AUDIO_DIR` and `LABELS_CSV` in `src/pipeline.py`.

---

### Running the pipeline

From the project root:

```bash
python -m src.pipeline
```

This will:

- Load the labels CSV and audio files.
- Extract three simple features per file:
  - zero-crossing rate
  - spectral centroid
  - spectral bandwidth
- Standardize the features with `StandardScaler`.
- Split the data into **train**, **validation**, and **test** sets.
- Print the shapes of the resulting arrays.

These outputs are intended to be consumed by future code in `src/models.py` for training and evaluation.

---

### Next steps

- Implement model training and evaluation logic in `src/models.py`.
- Add more robust feature extraction (e.g. MFCCs, log-mel spectrogram statistics).
- Introduce configuration (YAML/JSON) instead of hard-coded paths and split ratios.

