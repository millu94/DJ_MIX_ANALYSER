### AML Speakrightish

A small audio machine learning project for experimenting with basic preprocessing, dataset pipelines, and model training.

The current implementation focuses on:
- **Preprocessing**: simple, hand-crafted audio features (zero-crossing rate, spectral centroid, spectral bandwidth).
- **Pipeline**: reading class folders directly, auto-generating labels, and creating standardized train/validation/test splits.
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
   
   *ENVIRMONMENT NEEDED FOR DATASET_FORMAT*
   ## Environment Setup

To create a Conda environment for this project, follow these steps:

1. Install Miniconda or Anaconda from the official Conda website.
2. Create a new environment using the following command:
conda create --name dataset_format_env python=3.12
conda activate dataset_format-env
conda install -c conda-forge pydub ffmpeg

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

---

### Data layout

By default, the pipeline expects the following layout relative to the project root:

- `datasets/Main Phase - RA Mix, Good Transitions/` – contains `good_transition_*.wav`
- `datasets/Main Phase - RA Mix, Bad Transitions/` – contains `bad_transition_*.wav`

Labels are inferred automatically from folder names:

- Files in **Good Transitions** are labeled `1`
- Files in **Bad Transitions** are labeled `0`

You can change the root path by editing `DATASETS_ROOT` in `src/pipeline.py`.

test 12122121q

---

### Running the pipeline

From the project root:

```bash
python -m src.pipeline
```

This will:

- Load `.wav` files from the two class folders.
- Auto-generate binary labels (`good=1`, `bad=0`).
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

