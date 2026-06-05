// SplitAI — hovedlogikk i nettleseren.
//  * Laster Rust/WASM-backbone (CPU) og prøver WebGL2-backbone (GPU).
//  * Trener MLP-hodet lokalt (WASM), lagrer hode-vektene til serveren.
//  * Delt inferens: kjører backbone lokalt, sender kun features til serveren.

// Relative import-stier — virker både under Django (/) og GitHub Pages (/splitai/).
import init, { extract_features, MlpHead, feat_dim, input_size, backbone_weights_json }
  from "../wasm/splitai_wasm.js";
import { GpuBackbone } from "./gpu_backbone.js";
import { initDetectUi } from "./detect_ui.js";
import { makeBackend } from "./backend.js";

const $ = (id) => document.getElementById(id);
const IN = 48; // backbone-input (matcher Rust input_size())

let gpu = null;            // GpuBackbone eller null
let useGpu = false;        // aktiv modus
let trained = null;        // { head: MlpHead, classes: [..], n: int }
let samples = [];          // { canvas, features: Float32Array, labelEl }
let backend = null;        // ServerBackend | LocalBackend

// ---- bruker ("simulert") ---------------------------------------------------
function currentUser() { return localStorage.getItem("splitai_user") || ""; }
function setUser(name) {
  localStorage.setItem("splitai_user", name);
  $("current-user").textContent = name ? `bruker: ${name}` : "";
  refreshUsers();
}
async function refreshUsers() {
  try {
    const users = await backend.listUsers();
    $("user-list").textContent = users.length
      ? "Kjente brukere: " + users.map((u) => `${u.name} (${u.models})`).join(", ")
      : "Ingen brukere ennå.";
  } catch { /* ignorer */ }
}

// ---- backbone (felles inngang) --------------------------------------------
const work = document.createElement("canvas");
work.width = IN; work.height = IN;
const wctx = work.getContext("2d", { willReadFrequently: true });

function imageToRgba(img) {
  wctx.drawImage(img, 0, 0, IN, IN);
  return wctx.getImageData(0, 0, IN, IN).data; // Uint8ClampedArray IN*IN*4
}
function featuresFromRgba(rgba) {
  if (useGpu && gpu) return gpu.extract(rgba);
  return extract_features(rgba, IN, IN);
}
function featuresFromImage(img) { return featuresFromRgba(imageToRgba(img)); }

function loadImage(file) {
  return new Promise((res, rej) => {
    const img = new Image();
    img.onload = () => res(img);
    img.onerror = rej;
    img.src = URL.createObjectURL(file);
  });
}

// ---- GPU/CPU-selvtest ------------------------------------------------------
// CPU/WASM er fasiten alle klienter deler. En klient bruker bare GPU hvis dens
// egen GPU matcher CPU innen toleranse — slik forblir features kompatible på
// tvers av nettlesere uansett om de kjører GPU eller CPU.
function selfTest() {
  if (!gpu) { useGpu = false; $("selftest").textContent = "kun CPU (WASM)"; return; }
  const rgba = new Uint8ClampedArray(IN * IN * 4);
  for (let i = 0; i < rgba.length; i++) rgba[i] = (i * 73 + 11) & 255;
  const a = extract_features(rgba, IN, IN);
  const b = gpu.extract(rgba);
  let maxd = 0;
  for (let i = 0; i < a.length; i++) maxd = Math.max(maxd, Math.abs(a[i] - b[i]));
  const ok = maxd < 1e-3;
  useGpu = ok; // fall tilbake til CPU hvis GPU-en ikke matcher
  $("selftest").textContent = `maks avvik GPU↔CPU = ${maxd.toExponential(2)} (` +
    (ok ? "match ✔ — bruker GPU" : "avvik! — faller tilbake til CPU") + ")";
}

// ---- trening ---------------------------------------------------------------
function addSampleCard(img) {
  const wrap = document.createElement("div");
  wrap.className = "thumb";
  const c = document.createElement("canvas");
  c.width = IN; c.height = IN;
  c.getContext("2d").drawImage(img, 0, 0, IN, IN);
  const label = document.createElement("input");
  label.type = "text"; label.placeholder = "etikett";
  wrap.append(c, label);
  $("gallery").append(wrap);
  const features = featuresFromImage(img);
  samples.push({ canvas: c, features, labelEl: label });
  $("train-btn").disabled = false;
}

async function onFiles(ev) {
  for (const f of ev.target.files) {
    const img = await loadImage(f);
    addSampleCard(img);
  }
  ev.target.value = "";
}

// Brukes av YOLO-seksjonen: klipp ut en boks fra et bilde og legg den (med
// etikett) i treningssettet. Slik retrenes hodet på YOLO-annoterte utsnitt.
function addLabeledCrop(source, sx, sy, sw, sh, label) {
  const wrap = document.createElement("div");
  wrap.className = "thumb";
  const c = document.createElement("canvas");
  c.width = IN; c.height = IN;
  const cx = c.getContext("2d", { willReadFrequently: true });
  cx.drawImage(source, sx, sy, sw, sh, 0, 0, IN, IN);
  const labelEl = document.createElement("input");
  labelEl.type = "text"; labelEl.value = label || "";
  wrap.append(c, labelEl);
  $("gallery").append(wrap);
  const rgba = cx.getImageData(0, 0, IN, IN).data;
  const features = featuresFromRgba(rgba);
  samples.push({ canvas: c, features, labelEl });
  $("train-btn").disabled = false;
}

function trainHead() {
  const labeled = samples.filter((s) => s.labelEl.value.trim());
  if (labeled.length < 2) { $("train-out").textContent = "Trenger minst 2 merkede bilder."; return; }
  const classes = [];
  for (const s of labeled) {
    const l = s.labelEl.value.trim();
    if (!classes.includes(l)) classes.push(l);
  }
  if (classes.length < 2) { $("train-out").textContent = "Trenger minst 2 ulike etiketter."; return; }

  const hidden = Math.max(2, parseInt($("hidden").value, 10) || 24);
  const epochs = Math.max(10, parseInt($("epochs").value, 10) || 300);
  const lr = Math.max(0.001, parseFloat($("lr").value) || 0.3);

  const head = new MlpHead(classes.length, hidden);
  for (const s of labeled) head.add_sample(s.features, classes.indexOf(s.labelEl.value.trim()));

  let loss = 0;
  for (let e = 0; e < epochs; e++) loss = head.train_epoch(lr);

  // treningsnøyaktighet
  let correct = 0;
  for (const s of labeled) {
    const p = head.predict(s.features);
    let best = 0;
    for (let i = 1; i < p.length; i++) if (p[i] > p[best]) best = i;
    if (best === classes.indexOf(s.labelEl.value.trim())) correct++;
  }
  trained = { head, classes, n: labeled.length };
  $("train-out").innerHTML =
    `Trent på ${labeled.length} bilder, ${classes.length} klasser: ${classes.join(", ")}\n` +
    `Sluttap (cross-entropy): ${loss.toFixed(4)} · treningsnøyaktighet: ` +
    `${correct}/${labeled.length} (${Math.round(100 * correct / labeled.length)}%)`;
  $("save-btn").disabled = false;
}

async function saveModel() {
  if (!trained) return;
  if (!currentUser()) { alert("Velg bruker først (steg 1)."); return; }
  const body = {
    name: $("model-name").value.trim(),
    classes: trained.classes,
    weights: trained.head.export_json(),
    n_samples: trained.n,
  };
  const d = await backend.saveModel(body, currentUser());
  if (d.error) { $("train-out").textContent = "Feil: " + d.error; return; }
  const hvor = backend.kind === "server"
    ? "på serveren. Hode-vektene ligger nå kun på serveren."
    : "lokalt i nettleseren (localStorage). Vektene forlater aldri denne nettleseren.";
  $("train-out").textContent = `Lagret modell #${d.id} «${d.name}» ${hvor}`;
  refreshModels();
  refreshUsers();
}

// ---- delt inferens ---------------------------------------------------------
let selectedModel = null;

async function refreshModels() {
  const models = await backend.listModels();
  const box = $("model-list");
  box.innerHTML = "";
  if (!models.length) { box.innerHTML = "<p class='muted'>Ingen delte modeller ennå.</p>"; return; }
  for (const m of models) {
    const lab = document.createElement("label");
    const radio = document.createElement("input");
    radio.type = "radio"; radio.name = "model"; radio.value = m.id;
    radio.onchange = () => { selectedModel = m; $("infer-btn").disabled = false; };
    lab.append(radio);
    lab.append(document.createTextNode(
      `#${m.id} «${m.name}» — eier: ${m.owner} · klasser: [${m.classes.join(", ")}] · ${m.n_samples} bilder`));
    box.append(lab);
  }
}

async function runInference() {
  const file = $("infer-file").files[0];
  if (!file || !selectedModel) return;
  const img = await loadImage(file);
  const feat = featuresFromImage(img);
  const featArr = Array.from(feat);

  const d = await backend.infer(selectedModel.id, featArr, currentUser());
  if (d.error) { $("infer-out").textContent = "Feil: " + d.error; return; }

  const rows = d.predictions.map((p, i) => {
    const pct = Math.round(p.prob * 100);
    return `<div class="pred"><span class="name ${i === 0 ? "win" : ""}">${p.label}</span>` +
      `<span class="bar"><span style="width:${pct}%"></span></span><span>${pct}%</span></div>`;
  }).join("");
  const sent = featArr.map((v) => v.toFixed(3)).join(", ");
  const dest = backend.kind === "server" ? "sendt til server" : "brukt lokalt";
  $("infer-out").innerHTML =
    `<b>Svar fra modell «${d.model}» (eier ${d.owner}):</b>${rows}` +
    `<p class="muted small">${d.note}</p>` +
    `<details><summary class="mono">feature-vektor ${dest} (${featArr.length}-d)</summary>` +
    `<div class="mono">[${sent}]</div></details>`;
}

// ---- oppstart --------------------------------------------------------------
async function main() {
  await init();
  backend = await makeBackend();
  const bbJson = backbone_weights_json();
  gpu = GpuBackbone.tryCreate(JSON.parse(bbJson));
  selfTest(); // setter useGpu basert på om GPU matcher CPU
  $("backbone-mode").textContent = useGpu ? "WebGL2 GPU (fallback: WASM)" : "WASM CPU";

  const lagring = backend.kind === "server"
    ? "server (Django + sqlite)" : "lokalt i nettleseren (localStorage)";
  $("status").textContent =
    `Klar. Backbone: ${useGpu ? "GPU (WebGL2)" : "CPU (WASM)"} · ` +
    `input ${input_size()}×${input_size()} · feature-dim ${feat_dim()} · lagring: ${lagring}.`;

  // brukere
  $("user-name").value = currentUser();
  setUser(currentUser());
  $("user-set").onclick = () => {
    const n = $("user-name").value.trim();
    if (n) setUser(n);
  };

  // trening
  $("train-files").onchange = onFiles;
  $("apply-bulk").onclick = () => {
    const v = $("bulk-label").value.trim();
    if (v) samples.forEach((s) => { if (!s.labelEl.value.trim()) s.labelEl.value = v; });
  };
  $("train-btn").onclick = trainHead;
  $("save-btn").onclick = saveModel;

  // inferens
  $("refresh-models").onclick = refreshModels;
  $("infer-btn").onclick = runInference;
  refreshModels();

  // YOLO-seksjon (default-modell + live + annotering)
  const cocoNames = await fetch(
    new URL("../models/coco_names.json", import.meta.url)
  ).then((r) => r.json());
  initDetectUi({ addLabeledCrop }, cocoNames);
}

main().catch((e) => { $("status").textContent = "Feil ved oppstart: " + e; console.error(e); });
