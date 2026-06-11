"""Server-side YOLO tail: the last ~8 decode ops (model B).

The browser runs model A (the early layers) and sends the ~7 kB intermediate
tensors here; this module runs model B and returns the final detections in
640×640 letterbox space, plus the time spent in the model (server_ms).

This demonstrates split inference on the pretrained YOLO without any training:
the early part runs on the client, the final part runs on the server.
"""
from __future__ import annotations

import time

import numpy as np
from django.conf import settings

# Names of the cut tensors (must match static/js/yolo.js CUT and the extracted
# server_models/yolov10n_b.onnx inputs).
CUT = [
    "/model.23/GatherElements_output_0",  # boxes   [1,300,4] f32
    "/model.23/TopK_1_output_0",          # scores  [1,300]   f32
    "/model.23/TopK_1_output_1",          # indices [1,300]   i64
]

_SESSION = None


def _session():
    global _SESSION
    if _SESSION is None:
        import onnxruntime as ort  # imported lazily so the rest works without it

        path = settings.BASE_DIR / "server_models" / "yolov10n_b.onnx"
        _SESSION = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return _SESSION


def run_tail(boxes, scores, indices, score_thresh: float = 0.3):
    """Run model B. Returns (detections, server_ms).

    detections are in 640×640 letterbox space; the client maps them back.
    """
    b = np.asarray(boxes, dtype=np.float32).reshape(1, 300, 4)
    s = np.asarray(scores, dtype=np.float32).reshape(1, 300)
    i = np.asarray(indices, dtype=np.int64).reshape(1, 300)

    sess = _session()
    feeds = {CUT[0]: b, CUT[1]: s, CUT[2]: i}
    t0 = time.perf_counter()
    out = sess.run(["output0"], feeds)[0][0]  # (300, 6)
    server_ms = (time.perf_counter() - t0) * 1000.0

    dets = [
        {
            "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
            "score": float(sc), "cls": int(cl),
        }
        for x1, y1, x2, y2, sc, cl in out
        if sc >= score_thresh
    ]
    return dets, server_ms
