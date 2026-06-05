"""Server-side execution of the MLP head (the "final layers" of the funnel).

The backbone runs in the browser (Rust/WASM or WebGL). The client sends only the
feature vector here; the server runs the head weights and returns the answer.
This way the trained weights never leave the server.

The layout must match `MlpHead.export_json()` in wasm/src/lib.rs:
    w1: [hidden][feat]  (row-major)
    w2: [classes][hidden]
"""
from __future__ import annotations

import numpy as np


def run_head(weights: dict, feat: list[float]) -> np.ndarray:
    """Return the probability distribution (softmax) over the classes."""
    feat_dim = int(weights["feat"])
    hidden = int(weights["hidden"])
    classes = int(weights["classes"])

    x = np.asarray(feat, dtype=np.float32)
    if x.shape != (feat_dim,):
        raise ValueError(
            f"feature vector has length {x.shape[0]}, expected {feat_dim}"
        )

    w1 = np.asarray(weights["w1"], dtype=np.float32).reshape(hidden, feat_dim)
    b1 = np.asarray(weights["b1"], dtype=np.float32)
    w2 = np.asarray(weights["w2"], dtype=np.float32).reshape(classes, hidden)
    b2 = np.asarray(weights["b2"], dtype=np.float32)

    a1 = np.maximum(w1 @ x + b1, 0.0)  # ReLU
    z2 = w2 @ a1 + b2
    z2 -= z2.max()
    e = np.exp(z2)
    return e / e.sum()
