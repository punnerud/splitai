"""Server-side YOLO detection head (early-cut split, model B2).

The browser runs model A2 (backbone + neck, GPU-heavy) and sends the three neck
feature maps P3/P4/P5 as int8 (per-tensor symmetric quantization) — ~700 kB per
image. This module dequantizes them and runs the detection head (which holds real
conv weights), so the heavy front runs distributed on clients and the server only
does the lighter head work on CPU.

Wire format: request body = concatenated int8 bytes (P3, then P4, then P5);
the `X-Meta` header carries {shapes, scales, thresh}.
"""
from __future__ import annotations

import time

import numpy as np
from django.conf import settings

# Cut tensors (must match static/js/yolo.js CUT2 and the extracted b2 inputs).
CUT2 = [
    "/model.16/cv2/act/Mul_output_0",  # P3 [1,64,80,80]
    "/model.19/cv2/act/Mul_output_0",  # P4 [1,128,40,40]
    "/model.22/cv2/act/Mul_output_0",  # P5 [1,256,20,20]
]

_SESSION = None


def _session():
    global _SESSION
    if _SESSION is None:
        import onnxruntime as ort

        path = settings.BASE_DIR / "server_models" / "yolov10n_b2.onnx"
        _SESSION = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return _SESSION


def run_head(raw: bytes, shapes, scales, score_thresh: float = 0.3):
    """Dequantize the int8 feature maps and run the head. Returns
    (detections in 640-space, server_ms, bytes_used)."""
    feeds = {}
    off = 0
    for name, shape, scale in zip(CUT2, shapes, scales):
        n = int(np.prod(shape))
        q = np.frombuffer(raw, dtype=np.int8, count=n, offset=off)
        feeds[name] = (q.astype(np.float32) * float(scale)).reshape(shape)
        off += n

    sess = _session()
    t0 = time.perf_counter()
    out = sess.run(["output0"], feeds)[0][0]  # (300, 6)
    server_ms = (time.perf_counter() - t0) * 1000.0

    dets = [
        {"x1": float(a), "y1": float(b), "x2": float(c), "y2": float(d),
         "score": float(s), "cls": int(cl)}
        for a, b, c, d, s, cl in out
        if s >= score_thresh
    ]
    return dets, server_ms, off
