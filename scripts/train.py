from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from aml_speakrightish.data.io import load_labels
from aml_speakrightish.data.splits import stratified_split
from aml_speakrightish.features.audio_features import featurize_file
from aml_speakrightish.models.registry import create_model, predict_proba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--model", default="sklearn", choices=["sklearn", "pytorch"])
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--duration", type=float, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = load_labels(args.labels, args.audio_dir)
    y = np.array([e.label for e in examples], dtype=np.int64)

    X = np.stack(
        [featurize_file(e.path, sr=args.sr, duration=args.duration) for e in examples], axis=0
    )

    train_idx, test_idx = stratified_split(y, test_size=0.2, seed=42)
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    bundle = create_model(args.model, input_dim=X.shape[1])
    model = bundle.model

    model.fit(X_train, y_train)
    p = predict_proba(model, X_test)
    auc = float(roc_auc_score(y_test, p))

    # Save model
    Path("models").mkdir(parents=True, exist_ok=True)
    model.save(bundle.save_path)  # type: ignore[attr-defined]

    # Plot ROC
    fpr, tpr, _ = roc_curve(y_test, p)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC ({bundle.name})")
    plt.legend()
    roc_path = out_dir / f"roc_{bundle.name}.png"
    plt.savefig(roc_path, dpi=150, bbox_inches="tight")

    (out_dir / "metrics.txt").write_text(f"model={bundle.name}\nauc={auc:.6f}\n", encoding="utf-8")
    print(f"Saved model to {bundle.save_path}")
    print(f"AUC: {auc:.4f}")
    print(f"ROC: {roc_path}")


if __name__ == "__main__":
    main()

