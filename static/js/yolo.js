// Pretrained YOLOv10n (COCO) in the browser via onnxruntime-web.
// GPU (WebGPU) when supported, otherwise WASM. The model is NMS-free (YOLOv10
// end-to-end): output0 = [1, 300, 6] where each row is [x1, y1, x2, y2, score,
// class] in the 640×640 space (with letterbox padding).
//
// Two paths:
//   * Yolo       — runs the WHOLE model locally (full detection in the browser).
//   * YoloSplit  — runs only the early part (model A) locally and returns the
//                  ~7 kB intermediate tensors; the last ~8 decode ops (model B)
//                  run on the server (/api/yolo_tail). Lets you see real split
//                  inference + the server processing time, live.

// WASM-only build: the most reliable setup across all devices (incl. iPhone) and
// needs only a single .wasm file. (WebGPU would require ~38 MB of extra asyncify
// files; the backbone already uses WebGL GPU, so YOLO runs on WASM here.)
import * as ort from "../vendor/ort/ort.wasm.bundle.min.mjs";

// Relative to this module — works both under Django (/) and GitHub Pages (/splitai/).
ort.env.wasm.wasmPaths = new URL("../vendor/ort/", import.meta.url).href;
ort.env.wasm.numThreads = 1; // single-threaded → no SharedArrayBuffer/COOP-COEP needed

const MODEL_URL = new URL("../models/yolov10n_quant.onnx", import.meta.url).href;
const MODEL_A_URL = new URL("../models/yolov10n_a.onnx", import.meta.url).href;
const NAMES_URL = new URL("../models/coco_names.json", import.meta.url).href;
const SIZE = 640;

// Names of the cut tensors between model A (browser) and model B (server).
export const CUT = [
  "/model.23/GatherElements_output_0", // boxes  [1,300,4] f32
  "/model.23/TopK_1_output_0",         // scores [1,300]   f32
  "/model.23/TopK_1_output_1",         // indices[1,300]   i64
];

// Letterbox `source` into a SIZE×SIZE RGB float tensor (NCHW, [0,1]).
// Returns the tensor plus the params needed to map boxes back to source pixels.
function preprocess(source, cx) {
  const sw = source.videoWidth || source.naturalWidth || source.width;
  const sh = source.videoHeight || source.naturalHeight || source.height;
  const r = Math.min(SIZE / sw, SIZE / sh);
  const nw = Math.round(sw * r), nh = Math.round(sh * r);
  const padX = Math.floor((SIZE - nw) / 2), padY = Math.floor((SIZE - nh) / 2);
  cx.fillStyle = "rgb(114,114,114)";
  cx.fillRect(0, 0, SIZE, SIZE);
  cx.drawImage(source, 0, 0, sw, sh, padX, padY, nw, nh);
  const px = cx.getImageData(0, 0, SIZE, SIZE).data;
  const input = new Float32Array(3 * SIZE * SIZE);
  const plane = SIZE * SIZE;
  for (let i = 0; i < plane; i++) {
    input[i] = px[i * 4] / 255;
    input[plane + i] = px[i * 4 + 1] / 255;
    input[2 * plane + i] = px[i * 4 + 2] / 255;
  }
  return { tensor: new ort.Tensor("float32", input, [1, 3, SIZE, SIZE]), r, padX, padY, sw, sh };
}

// Map [x1,y1,x2,y2] in 640-letterbox space back to a source-pixel box.
export function unletterbox(x1, y1, x2, y2, lb) {
  const bx1 = Math.max(0, (x1 - lb.padX) / lb.r);
  const by1 = Math.max(0, (y1 - lb.padY) / lb.r);
  const bx2 = Math.min(lb.sw, (x2 - lb.padX) / lb.r);
  const by2 = Math.min(lb.sh, (y2 - lb.padY) / lb.r);
  return { x: bx1, y: by1, w: bx2 - bx1, h: by2 - by1 };
}

function make640Canvas() {
  const cv = document.createElement("canvas");
  cv.width = SIZE; cv.height = SIZE;
  return cv.getContext("2d", { willReadFrequently: true });
}

export class Yolo {
  static async load() {
    const names = await fetch(NAMES_URL).then((r) => r.json());
    const session = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ["wasm"],
    });
    return new Yolo(session, names, "wasm");
  }

  constructor(session, names, ep) {
    this.session = session;
    this.names = names;
    this.ep = ep;
    this.inputName = session.inputNames[0]; // "images"
    this.outputName = session.outputNames[0]; // "output0"
    this._cx = make640Canvas();
  }

  // source: HTMLVideoElement | HTMLCanvasElement | HTMLImageElement
  // Returns boxes in the source's pixel coordinates.
  async detect(source, scoreThresh = 0.3) {
    const sw = source.videoWidth || source.naturalWidth || source.width;
    const sh = source.videoHeight || source.naturalHeight || source.height;
    if (!sw || !sh) return { boxes: [], sw: 0, sh: 0 };

    const lb = preprocess(source, this._cx);
    const out = await this.session.run({ [this.inputName]: lb.tensor });
    const d = out[this.outputName].data; // 300×6

    const boxes = [];
    for (let i = 0; i < 300; i++) {
      const o = i * 6;
      const score = d[o + 4];
      if (score < scoreThresh) continue;
      const b = unletterbox(d[o], d[o + 1], d[o + 2], d[o + 3], lb);
      const cls = d[o + 5] | 0;
      boxes.push({ ...b, score, cls, label: this.names[cls] || `cls${cls}` });
    }
    return { boxes, sw, sh };
  }
}

// Runs only model A (early layers) locally; the server finishes the decode.
export class YoloSplit {
  static async load() {
    const session = await ort.InferenceSession.create(MODEL_A_URL, {
      executionProviders: ["wasm"],
    });
    return new YoloSplit(session);
  }

  constructor(session) {
    this.session = session;
    this._cx = make640Canvas();
  }

  // Run the early layers and return the small intermediate tensors + letterbox
  // params + the local inference time (ms).
  async runHalf(source) {
    const t0 = performance.now();
    const lb = preprocess(source, this._cx);
    const out = await this.session.run({ images: lb.tensor });
    const localMs = performance.now() - t0;
    return {
      boxes: Array.from(out[CUT[0]].data),          // 1200 floats
      scores: Array.from(out[CUT[1]].data),         // 300 floats
      indices: Array.from(out[CUT[2]].data, Number), // 300 ints (from BigInt64)
      lb, localMs,
    };
  }
}
