# AML_Speakrightish

Binary classifier that predicts whether an audio mix is **good** or **bad** (labels: `0/1`) from a small curated dataset (~100 mixes).

## Repo layout

- `data/`
  - `raw/`: your original audio files + labels CSV (not committed)
  - `processed/`: cached intermediate artifacts (not committed)
- `src/aml_speakrightish/`: reusable pipeline code (split, preprocess, features, models)
- `scripts/`: command-line entrypoints (train/eval)
- `tests/`: quick checks / smoke tests
- `outputs/`: plots + reports (not committed)
- `models/`: trained models (not committed)

## Dataset format (expected)

Put your files here:

- `data/raw/audio/` — audio files (e.g. `mix_001.wav`, `mix_002.wav`, ...)
- `data/raw/labels.csv` — two columns:
  - `file`: filename relative to `data/raw/audio/` (e.g. `mix_001.wav`)
  - `label`: `0` (bad) or `1` (good)

Example `labels.csv`:

```csv
file,label
mix_001.wav,1
mix_002.wav,0
```

## Setup

Create a virtualenv and install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Train and evaluate

Train a baseline model (sklearn):

```bash
python scripts/train.py --labels data/raw/labels.csv --audio-dir data/raw/audio --out-dir outputs --model sklearn
```

Train a neural net model (PyTorch):

```bash
python scripts/train.py --labels data/raw/labels.csv --audio-dir data/raw/audio --out-dir outputs --model pytorch
```

Evaluate a saved model:

```bash
python scripts/eval.py --labels data/raw/labels.csv --audio-dir data/raw/audio --model-path models/model.joblib
```

## Notes / next steps

- Start simple: MFCC/log-mel features + a linear model (logistic regression / linear SVM).
- With only ~100 examples, use stratified splits + cross-validation and keep preprocessing consistent between train/test.

