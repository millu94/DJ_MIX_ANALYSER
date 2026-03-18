from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, roc_auc_score

from aml_speakrightish.data.io import load_labels
from aml_speakrightish.features.audio_features import featurize_file
from aml_speakrightish.models.sklearn_model import SklearnBinaryClassifier
from aml_speakrightish.models.torch_model import TorchBinaryClassifier


def load_model(model_path: str):
    p = Path(model_path)
    if p.suffix == ".joblib":
        return SklearnBinaryClassifier.load(str(p))
    return TorchBinaryClassifier.load(str(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--duration", type=float, default=None)
    args = ap.parse_args()

    examples = load_labels(args.labels, args.audio_dir)
    y = np.array([e.label for e in examples], dtype=np.int64)
    X = np.stack(
        [featurize_file(e.path, sr=args.sr, duration=args.duration) for e in examples], axis=0
    )

    model = load_model(args.model_path)
    p = model.predict_proba(X)
    auc = float(roc_auc_score(y, p))
    yhat = (p >= 0.5).astype(np.int64)

    print(f"AUC: {auc:.4f}")
    print(classification_report(y, yhat, digits=4))


if __name__ == "__main__":
    main()

