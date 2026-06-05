"""Server-side kjoring av MLP-hodet (de "siste lagene" i trakten).

Backbone-en kjorer i nettleseren (Rust/WASM eller WebGL). Klienten sender bare
feature-vektoren hit; serveren kjorer hode-vektene og returnerer svaret. Slik
forlater de trente vektene aldri serveren.

Layout ma matche `MlpHead.export_json()` i wasm/src/lib.rs:
    w1: [hidden][feat]  (rad-major)
    w2: [classes][hidden]
"""
from __future__ import annotations

import numpy as np


def run_head(weights: dict, feat: list[float]) -> np.ndarray:
    """Returner sannsynlighetsfordeling (softmax) over klassene."""
    feat_dim = int(weights["feat"])
    hidden = int(weights["hidden"])
    classes = int(weights["classes"])

    x = np.asarray(feat, dtype=np.float32)
    if x.shape != (feat_dim,):
        raise ValueError(
            f"feature-vektor har lengde {x.shape[0]}, forventet {feat_dim}"
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
