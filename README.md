# SplitAI

A local Django project that demonstrates a **shared "funnel" model** where the
early layers (the backbone) run in the browser via **Rust → WASM** (with
**WebGL2 GPU** when supported), and the final layers (the head) are trained
locally and run shared on the server — without revealing the trained weights to
other users.

```
  Browser (user A)                             Server (Django + numpy)
  ┌────────────────────────────┐               ┌──────────────────────────┐
  │ image → backbone (WASM/GPU) │  features     │   head (final layers)    │
  │   fixed, shared weights     │ ───────────▶  │   secret, trained        │
  │   → 32-d feature vector      │              │   weights → answer       │
  └────────────────────────────┘               └──────────────────────────┘
        ▲ trains the head locally                     ▲ stores the head
        └── exports head weights ──────────────────────┘
```

## Idea

* **Backbone (early layers):** conv layers with **fixed, deterministic weights**
  that are the same for all clients. Runs in the browser. Produces a 32-dim
  feature vector. The heavy part of the "funnel", but frozen.
* **Head (final layers):** a small MLP. This is **trained** via transfer learning
  in the browser (on frozen features) and is the "valuable/secret" part.
* **Sharing without revealing weights:** the head weights are stored only on the
  server. Other users run the backbone locally on *their* images and send **only
  the feature vector** to the server, which runs the head and returns the answer.
  The weights never leave the server.
* **GPU via WebGL2:** the backbone runs on the GPU when the browser supports
  WebGL2 + `EXT_color_buffer_float`, otherwise it falls back to CPU/WASM. A
  built-in self-test checks that GPU and CPU features match (CPU is the ground
  truth everyone shares); a client only uses the GPU if its own test matches.

> **Caveat:** This is a compact YOLO-*inspired* CNN (conv backbone + MLP head),
> not a full YOLO. The point is to show the split/funnel pipeline end to end. The
> architecture in `wasm/src/lib.rs` can be swapped for a larger/real backbone
> later — the server and frontend flow are unchanged as long as the feature
> dimension comes along.

## Requirements

* Python 3.12 (the venv lives in `venv/`)
* Rust + `wasm-pack` (only to *rebuild* WASM — a prebuilt copy is in `static/wasm/`)

## Getting started

Plain HTTP (everything except the webcam):

```bash
./run.sh                      # migrates + starts http://127.0.0.1:8000/
```

**With the webcam (HTTPS, works on phones over the LAN):**

```bash
./run_https.sh                # → https://<your-LAN-IP>:8443/  (self-signed cert)
```

The webcam (`getUserMedia`) requires a "secure context", so the live stream only
works on `https://` or `localhost` — hence the HTTPS variant. Accept the
certificate warning once per device. (`run_https.sh [port]` for another port.)

or manually:

```bash
python3.12 -m venv venv
./venv/bin/pip install -r requirements.txt
./build_wasm.sh               # builds Rust → static/wasm/  (optional, already built)
./venv/bin/python manage.py migrate
./venv/bin/python manage.py runserver
```

## Running browser-only (GitHub Pages)

The app also runs **completely without a server**. `static/js/backend.js` chooses
automatically:

* **ServerBackend** when the Django API responds (the main engine — models in sqlite).
* **LocalBackend** otherwise (e.g. GitHub Pages) — models are stored in
  `localStorage` and the head runs locally in the browser (`runHead` mirrors
  `core/head_runtime.py`). The head weights are only a few kB, so they fit
  locally with ease.

All paths are relative, so the same `index.html` works both under Django (`/`)
and under Pages (`/splitai/`). GitHub Pages provides HTTPS, so **the webcam works
there without any certificate setup**.

Enable Pages: repo → Settings → Pages → "Deploy from a branch" → `main` / `/ (root)`.
Then: `https://punnerud.github.io/splitai/`. (The status line in the app shows
whether storage is "server" or "localStorage".) Note: in Pages mode models are
not shared between devices — `localStorage` is per browser. For cross-device
sharing, run Django.

## Default model: YOLO (live detection → annotate → retrain)

Section 0 of the app runs a **pretrained YOLOv10n** (COCO, 80 classes incl.
person, cup, bottle …) in the browser via **onnxruntime-web** (WebGPU when
supported, otherwise WASM — everything vendored locally in `static/vendor/ort/` +
`static/models/`).

Workflow:
1. *Start camera* (HTTPS) or *upload image* → YOLO draws boxes live.
2. *Capture* → the frame is frozen and YOLO proposes boxes.
3. **Annotate:** drag to draw new boxes, drag/click to move/select, fix labels or
   delete in the list.
4. *Add boxes to training set* → each crop is cut out, run through the backbone
   (WASM/GPU) into a feature vector, and lands in step 2.
5. *Train the head* (step 2) → *Save the head* → share/run as usual.

> **"Retrain" = the head, not YOLO.** YOLO stays frozen as a box detector. What
> actually gets retrained is the lightweight MLP head (transfer learning on the
> annotated crops). True end-to-end YOLO training in the browser is not done — it
> requires full detection backprop and is unrealistic on the client side.

## Testing multiple users

No login. Each browser picks a name (stored in `localStorage`). Open the app in
**two different browsers or profiles**:

1. **User A (e.g. "alice"):** pick a name → upload labeled images → *Train the head*
   → *Save the head*.
2. **User B (e.g. "bob"):** pick a name → *Refresh model list* → select alice's
   model → upload an image → *Run*. The backbone runs locally on B's side; only
   the feature vector is sent, and alice's head weights stay on the server.

## Files

| Path | What |
|---|---|
| `wasm/src/lib.rs` | Rust: backbone (`extract_features`) + MLP head (`MlpHead`) |
| `static/js/gpu_backbone.js` | WebGL2 GPU version of the backbone (identical weights) |
| `static/js/yolo.js` | YOLOv10n via onnxruntime-web (pre/post-processing) |
| `static/js/detect_ui.js` | Webcam, live detection, annotation, → training set |
| `static/js/app.js` | Frontend: training, saving, shared inference, self-test |
| `static/models/`, `static/vendor/ort/` | YOLO model + onnxruntime-web (vendored) |
| `core/head_runtime.py` | numpy execution of the head on the server (matches Rust) |
| `core/views.py` | API: `/api/users`, `/api/models`, `/api/infer` |
| `run_https.sh` | HTTPS server (self-signed cert) for the webcam over the LAN |

## Tests

```bash
cd wasm && cargo test            # backbone normalization + head convergence
```

The numpy head (server) matches Rust's `predict` to ~1e-9 on the same
weights/features.
