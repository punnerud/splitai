// YOLO-seksjonen: live webkamera, deteksjon, annotering, og "legg bokser i
// treningssett" (som retrener det lette hodet via eksisterende backbone).
//
// Kobles til app.js via hooks: { addLabeledCrop(sourceCanvas, x,y,w,h, label) }.

import { Yolo } from "/static/js/yolo.js";

const $ = (id) => document.getElementById(id);

let yolo = null;
let hooks = null;

const frame = document.createElement("canvas"); // fryst bilde i native oppløsning
const fctx = frame.getContext("2d", { willReadFrequently: true });

let stream = null;
let streaming = false;
let busy = false;
let annoBoxes = []; // {x,y,w,h,label,cls,score} i frame-piksler
let selected = -1;

let mode = null; // 'draw' | 'move'
let dragStart = null; // {px,py}
let moveOff = null; // {dx,dy}

const PALETTE = ["#5b8cff", "#36c98a", "#ff6b6b", "#ffd166", "#b794f6", "#4dd0e1"];

function status(msg) { $("yolo-status").textContent = msg; }
function thresh() {
  const v = parseFloat($("yolo-thresh").value);
  return isNaN(v) ? 0.3 : Math.min(0.95, Math.max(0.05, v));
}

async function ensureYolo() {
  if (yolo) return yolo;
  status("laster YOLO-modell … (første gang ~26 MB)");
  yolo = await Yolo.load();
  status(`YOLO klar — kjører på ${yolo.ep}`);
  return yolo;
}

// ---- live webkamera --------------------------------------------------------
async function startCam() {
  try {
    await ensureYolo();
  } catch (e) { status("Kunne ikke laste YOLO: " + e); return; }

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    status("Webkamera krever HTTPS (secure context). Bruk HTTPS-serveren, eller last opp bilde i stedet.");
    return;
  }
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" } }, audio: false,
    });
  } catch (e) {
    status("Fikk ikke kameratilgang: " + e.message + " (på telefon kreves HTTPS).");
    return;
  }
  const video = $("cam");
  video.srcObject = stream;
  video.style.display = "block";
  await video.play();
  streaming = true;
  $("cam-capture").disabled = false;
  $("cam-start").textContent = "Stopp kamera";
  liveLoop();
}

function stopCam() {
  streaming = false;
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
  $("cam").style.display = "none";
  $("live").style.display = "none";
  $("cam-capture").disabled = true;
  $("cam-start").textContent = "Start kamera";
}

async function liveLoop() {
  const video = $("cam");
  if (!streaming) return;
  if (!busy && video.videoWidth) {
    busy = true;
    try {
      const { boxes, sw } = await yolo.detect(video, thresh());
      drawLive(boxes, sw);
    } catch (e) { console.error(e); }
    busy = false;
  }
  requestAnimationFrame(liveLoop);
}

function drawLive(boxes, sw) {
  const video = $("cam");
  const live = $("live");
  const cw = video.clientWidth, ch = video.clientHeight;
  if (live.width !== cw || live.height !== ch) { live.width = cw; live.height = ch; }
  live.style.display = "block";
  const s = cw / sw;
  const ctx = live.getContext("2d");
  ctx.clearRect(0, 0, cw, ch);
  ctx.lineWidth = 2;
  ctx.font = "14px sans-serif";
  ctx.textBaseline = "top";
  for (const b of boxes) {
    const c = PALETTE[b.cls % PALETTE.length];
    ctx.strokeStyle = c;
    ctx.strokeRect(b.x * s, b.y * s, b.w * s, b.h * s);
    const t = `${b.label} ${(b.score * 100) | 0}%`;
    ctx.fillStyle = c;
    const tw = ctx.measureText(t).width + 6;
    ctx.fillRect(b.x * s, b.y * s - 16, tw, 16);
    ctx.fillStyle = "#0b0d16";
    ctx.fillText(t, b.x * s + 3, b.y * s - 15);
  }
}

// ---- ta bilde / last opp → annoter ----------------------------------------
async function detectInto(source) {
  const sw = source.videoWidth || source.naturalWidth || source.width;
  const sh = source.videoHeight || source.naturalHeight || source.height;
  frame.width = sw; frame.height = sh;
  fctx.drawImage(source, 0, 0, sw, sh);
  let boxes = [];
  try {
    await ensureYolo();
    ({ boxes } = await yolo.detect(frame, thresh()));
  } catch (e) { status("YOLO-feil: " + e); }
  annoBoxes = boxes.map((b) => ({ ...b }));
  selected = annoBoxes.length ? 0 : -1;
  $("annotate").style.display = "block";
  setupAnnoCanvas();
  renderBoxList();
  drawAnno();
}

function captureFrame() {
  const video = $("cam");
  if (!video.videoWidth) return;
  detectInto(video);
}

async function uploadToYolo(file) {
  const img = new Image();
  img.onload = () => detectInto(img);
  img.src = URL.createObjectURL(file);
}

// ---- annoterings-canvas ----------------------------------------------------
function setupAnnoCanvas() {
  const anno = $("anno");
  anno.width = frame.width; anno.height = frame.height;
}

function drawAnno() {
  const anno = $("anno");
  const ctx = anno.getContext("2d");
  ctx.drawImage(frame, 0, 0);
  const fs = Math.max(14, frame.width / 45);
  ctx.font = `${fs}px sans-serif`;
  ctx.textBaseline = "top";
  ctx.lineWidth = Math.max(2, frame.width / 300);
  annoBoxes.forEach((b, i) => {
    const c = PALETTE[(b.cls ?? i) % PALETTE.length];
    ctx.strokeStyle = i === selected ? "#fff" : c;
    ctx.lineWidth = i === selected ? Math.max(3, frame.width / 200) : Math.max(2, frame.width / 350);
    ctx.strokeRect(b.x, b.y, b.w, b.h);
    const t = b.score ? `${b.label} ${(b.score * 100) | 0}%` : b.label;
    ctx.fillStyle = c;
    const tw = ctx.measureText(t).width + 8;
    ctx.fillRect(b.x, b.y - fs - 4, tw, fs + 4);
    ctx.fillStyle = "#0b0d16";
    ctx.fillText(t, b.x + 4, b.y - fs - 2);
  });
}

function evtToFrame(e) {
  const anno = $("anno");
  const rect = anno.getBoundingClientRect();
  const sx = anno.width / rect.width, sy = anno.height / rect.height;
  return { px: (e.clientX - rect.left) * sx, py: (e.clientY - rect.top) * sy };
}

function hitTest(px, py) {
  for (let i = annoBoxes.length - 1; i >= 0; i--) {
    const b = annoBoxes[i];
    if (px >= b.x && px <= b.x + b.w && py >= b.y && py <= b.y + b.h) return i;
  }
  return -1;
}

function onDown(e) {
  e.preventDefault();
  const { px, py } = evtToFrame(e);
  const hit = hitTest(px, py);
  if (hit >= 0) {
    selected = hit; mode = "move";
    moveOff = { dx: px - annoBoxes[hit].x, dy: py - annoBoxes[hit].y };
  } else {
    mode = "draw"; dragStart = { px, py };
    annoBoxes.push({ x: px, y: py, w: 0, h: 0, label: ($("new-label").value.trim() || "objekt") });
    selected = annoBoxes.length - 1;
  }
  $("anno").setPointerCapture(e.pointerId);
  renderBoxList(); drawAnno();
}

function onMove(e) {
  if (!mode) return;
  const { px, py } = evtToFrame(e);
  const b = annoBoxes[selected];
  if (mode === "draw") {
    b.x = Math.min(dragStart.px, px); b.y = Math.min(dragStart.py, py);
    b.w = Math.abs(px - dragStart.px); b.h = Math.abs(py - dragStart.py);
  } else if (mode === "move") {
    b.x = Math.max(0, Math.min(frame.width - b.w, px - moveOff.dx));
    b.y = Math.max(0, Math.min(frame.height - b.h, py - moveOff.dy));
  }
  drawAnno();
}

function onUp(e) {
  if (mode === "draw") {
    const b = annoBoxes[selected];
    if (b.w < 6 || b.h < 6) { annoBoxes.splice(selected, 1); selected = -1; }
  }
  mode = null; dragStart = null; moveOff = null;
  renderBoxList(); drawAnno();
}

// ---- boks-liste ------------------------------------------------------------
function renderBoxList() {
  const box = $("box-list");
  box.innerHTML = "";
  if (!annoBoxes.length) { box.innerHTML = "<p class='muted small'>Ingen bokser. Dra på bildet for å tegne en.</p>"; return; }
  annoBoxes.forEach((b, i) => {
    const row = document.createElement("div");
    row.className = "pred";
    const sw = document.createElement("span");
    sw.style.width = "14px"; sw.style.height = "14px"; sw.style.borderRadius = "3px";
    sw.style.background = PALETTE[(b.cls ?? i) % PALETTE.length];
    sw.style.outline = i === selected ? "2px solid #fff" : "none";
    const label = document.createElement("input");
    label.type = "text"; label.value = b.label; label.setAttribute("list", "coco-list");
    label.style.flex = "1";
    label.oninput = () => { b.label = label.value; };
    label.onfocus = () => { selected = i; drawAnno(); renderBoxListSelection(); };
    const sc = document.createElement("span");
    sc.className = "muted small";
    sc.textContent = b.score ? `${(b.score * 100) | 0}%` : "egen";
    const del = document.createElement("button");
    del.className = "secondary"; del.textContent = "slett";
    del.onclick = () => { annoBoxes.splice(i, 1); selected = -1; renderBoxList(); drawAnno(); };
    row.append(sw, label, sc, del);
    box.append(row);
  });
}
function renderBoxListSelection() {
  [...$("box-list").children].forEach((row, i) => {
    const sw = row.querySelector("span");
    if (sw) sw.style.outline = i === selected ? "2px solid #fff" : "none";
  });
}

// ---- legg i treningssett ---------------------------------------------------
function addToTrain() {
  let n = 0;
  for (const b of annoBoxes) {
    const label = (b.label || "").trim();
    if (!label) continue;
    const x = Math.max(0, Math.round(b.x)), y = Math.max(0, Math.round(b.y));
    const w = Math.min(frame.width - x, Math.round(b.w));
    const h = Math.min(frame.height - y, Math.round(b.h));
    if (w < 4 || h < 4) continue;
    hooks.addLabeledCrop(frame, x, y, w, h, label);
    n++;
  }
  status(`La til ${n} utsnitt i treningssettet (se steg 2 — Tren hodet).`);
}

// ---- init ------------------------------------------------------------------
export function initDetectUi(h, cocoNames) {
  hooks = h;
  // datalist for etikett-autofullfør
  const dl = document.createElement("datalist");
  dl.id = "coco-list";
  for (const name of cocoNames) {
    const o = document.createElement("option"); o.value = name; dl.append(o);
  }
  document.body.append(dl);

  $("cam-start").onclick = () => (streaming ? stopCam() : startCam());
  $("cam-capture").onclick = captureFrame;
  $("yolo-upload").onchange = (e) => { if (e.target.files[0]) uploadToYolo(e.target.files[0]); e.target.value = ""; };
  $("add-to-train").onclick = addToTrain;
  $("anno-clear").onclick = () => { annoBoxes = []; selected = -1; renderBoxList(); drawAnno(); };

  const anno = $("anno");
  anno.addEventListener("pointerdown", onDown);
  anno.addEventListener("pointermove", onMove);
  anno.addEventListener("pointerup", onUp);
}
