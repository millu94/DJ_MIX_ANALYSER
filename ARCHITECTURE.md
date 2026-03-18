# Architecture

This project is a simple end-to-end audio classification pipeline:

1. Read `labels.csv` + audio files
2. Create a stratified train/val split
3. Extract features (e.g. log-mel spectrogram summaries)
4. Train a model:
   - **sklearn** baseline (fast, good for small data)
   - **tensorflow** neural net (Keras)
5. Evaluate and write outputs (metrics + plots)

## Key modules

- `src/aml_speakrightish/data/`
  - `io.py`: load `labels.csv`, resolve audio paths
  - `splits.py`: stratified split utilities
- `src/aml_speakrightish/features/`
  - `audio_features.py`: log-mel / MFCC extraction helpers
- `src/aml_speakrightish/models/`
  - `sklearn_model.py`: baseline model + save/load
  - `torch_model.py`: PyTorch MLP model + save/load
  - `registry.py`: choose model by name (`sklearn` / `pytorch`)
- `scripts/`
  - `train.py`: CLI training entrypoint
  - `eval.py`: CLI evaluation entrypoint

