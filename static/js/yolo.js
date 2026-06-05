// Forhåndstrent YOLOv10n (COCO) i nettleseren via onnxruntime-web.
// GPU (WebGPU) når støttet, ellers WASM. Modellen er NMS-fri (YOLOv10 ende-til-
// ende): output0 = [1, 300, 6] der hver rad er [x1, y1, x2, y2, score, klasse]
// i 640×640-rommet (med letterbox-padding).

import * as ort from "../vendor/ort/ort.webgpu.bundle.min.mjs";

// Relativt til denne modulen — virker både under Django (/) og GitHub Pages (/splitai/).
ort.env.wasm.wasmPaths = new URL("../vendor/ort/", import.meta.url).href;
ort.env.wasm.numThreads = 1; // unngår SharedArrayBuffer/COOP-COEP-krav

const MODEL_URL = new URL("../models/yolov10n_quant.onnx", import.meta.url).href;
const NAMES_URL = new URL("../models/coco_names.json", import.meta.url).href;
const SIZE = 640;

export class Yolo {
  static async load() {
    const names = await fetch(NAMES_URL).then((r) => r.json());
    const wantGpu = typeof navigator !== "undefined" && !!navigator.gpu;
    const providers = wantGpu ? ["webgpu", "wasm"] : ["wasm"];
    let session, ep;
    try {
      session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: providers });
      ep = wantGpu ? "webgpu→wasm" : "wasm";
    } catch (e) {
      // fall tilbake til ren WASM hvis WebGPU-init feilet
      session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: ["wasm"] });
      ep = "wasm";
    }
    return new Yolo(session, names, ep);
  }

  constructor(session, names, ep) {
    this.session = session;
    this.names = names;
    this.ep = ep;
    this.inputName = session.inputNames[0]; // "images"
    this.outputName = session.outputNames[0]; // "output0"
    this._cv = document.createElement("canvas");
    this._cv.width = SIZE; this._cv.height = SIZE;
    this._cx = this._cv.getContext("2d", { willReadFrequently: true });
  }

  // source: HTMLVideoElement | HTMLCanvasElement | HTMLImageElement
  // Returnerer bokser i kildens pikselkoordinater.
  async detect(source, scoreThresh = 0.3) {
    const sw = source.videoWidth || source.naturalWidth || source.width;
    const sh = source.videoHeight || source.naturalHeight || source.height;
    if (!sw || !sh) return { boxes: [], sw: 0, sh: 0 };

    // letterbox til 640×640 (behold aspekt, grå padding 114)
    const r = Math.min(SIZE / sw, SIZE / sh);
    const nw = Math.round(sw * r), nh = Math.round(sh * r);
    const padX = Math.floor((SIZE - nw) / 2), padY = Math.floor((SIZE - nh) / 2);
    this._cx.fillStyle = "rgb(114,114,114)";
    this._cx.fillRect(0, 0, SIZE, SIZE);
    this._cx.drawImage(source, 0, 0, sw, sh, padX, padY, nw, nh);
    const px = this._cx.getImageData(0, 0, SIZE, SIZE).data;

    // RGBA uint8 → planar RGB float [0,1], NCHW
    const input = new Float32Array(3 * SIZE * SIZE);
    const plane = SIZE * SIZE;
    for (let i = 0; i < plane; i++) {
      input[i] = px[i * 4] / 255;
      input[plane + i] = px[i * 4 + 1] / 255;
      input[2 * plane + i] = px[i * 4 + 2] / 255;
    }

    const tensor = new ort.Tensor("float32", input, [1, 3, SIZE, SIZE]);
    const out = await this.session.run({ [this.inputName]: tensor });
    const d = out[this.outputName].data; // 300×6

    const boxes = [];
    for (let i = 0; i < 300; i++) {
      const o = i * 6;
      const score = d[o + 4];
      if (score < scoreThresh) continue;
      const x1 = (d[o] - padX) / r;
      const y1 = (d[o + 1] - padY) / r;
      const x2 = (d[o + 2] - padX) / r;
      const y2 = (d[o + 3] - padY) / r;
      const cls = d[o + 5] | 0;
      boxes.push({
        x: Math.max(0, x1), y: Math.max(0, y1),
        w: Math.min(sw, x2) - Math.max(0, x1),
        h: Math.min(sh, y2) - Math.max(0, y1),
        score, cls, label: this.names[cls] || `cls${cls}`,
      });
    }
    return { boxes, sw, sh };
  }
}
