// Backend abstraction. Two implementations with the same interface:
//   * ServerBackend — talks to the Django API (the main engine).
//   * LocalBackend  — runs entirely in the browser and stores models in
//                     localStorage. Used on GitHub Pages / when no server is
//                     available.
//
// Interface (all async):
//   listUsers()                         -> [{name, models}]
//   listModels()                        -> [{id,name,owner,classes,feat_dim,hidden,n_samples,created}]
//   saveModel({name,classes,weights,n_samples}, user)  -> {id,name} | {error}
//   infer(modelId, features, user)      -> {model,owner,predictions:[{label,prob}],note} | {error}
//
// `weights` is the JSON string from MlpHead.export_json().

function api(path) {
  // Relative to the page URL: becomes /api/... under Django.
  return new URL(path, document.baseURI).href;
}

class ServerBackend {
  get kind() { return "server"; }

  async listUsers() {
    const r = await fetch(api("api/users"));
    return (await r.json()).users;
  }
  async listModels() {
    const r = await fetch(api("api/models"));
    return (await r.json()).models;
  }
  async saveModel(payload, user) {
    const r = await fetch(api("api/models"), {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-User": user },
      body: JSON.stringify(payload),
    });
    return r.json();
  }
  async infer(modelId, features, user) {
    const r = await fetch(api("api/infer"), {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-User": user },
      body: JSON.stringify({ model_id: modelId, features }),
    });
    return r.json();
  }
}

// Run the head (the final layers) locally in the browser. Mirrors
// core/head_runtime.py and the Rust MlpHead: w1 = [hidden][feat],
// w2 = [classes][hidden].
function runHead(w, x) {
  const F = w.feat, H = w.hidden, C = w.classes;
  const a1 = new Float32Array(H);
  for (let h = 0; h < H; h++) {
    let s = w.b1[h];
    for (let f = 0; f < F; f++) s += w.w1[h * F + f] * x[f];
    a1[h] = s > 0 ? s : 0; // ReLU
  }
  const z = new Float32Array(C);
  let mx = -Infinity;
  for (let c = 0; c < C; c++) {
    let s = w.b2[c];
    for (let h = 0; h < H; h++) s += w.w2[c * H + h] * a1[h];
    z[c] = s; if (s > mx) mx = s;
  }
  let sum = 0;
  for (let c = 0; c < C; c++) { z[c] = Math.exp(z[c] - mx); sum += z[c]; }
  for (let c = 0; c < C; c++) z[c] /= sum;
  return z;
}

const LS_KEY = "splitai_models";

class LocalBackend {
  get kind() { return "local"; }

  _load() {
    try { return JSON.parse(localStorage.getItem(LS_KEY)) || []; }
    catch { return []; }
  }
  _save(models) { localStorage.setItem(LS_KEY, JSON.stringify(models)); }

  async listUsers() {
    const counts = {};
    for (const m of this._load()) counts[m.owner] = (counts[m.owner] || 0) + 1;
    const cur = localStorage.getItem("splitai_user");
    if (cur && !(cur in counts)) counts[cur] = 0;
    return Object.entries(counts).map(([name, models]) => ({ name, models }));
  }
  async listModels() {
    // return metadata without leaking the weights (same contract as the server)
    return this._load().map(({ weights, ...meta }) => meta);
  }
  async saveModel(payload, user) {
    let weights;
    try { weights = JSON.parse(payload.weights); }
    catch (e) { return { error: "invalid weights" }; }
    if (payload.classes.length !== weights.classes) {
      return { error: "number of class names does not match the weights" };
    }
    const models = this._load();
    const id = models.reduce((m, x) => Math.max(m, x.id), 0) + 1;
    const now = new Date().toISOString();
    models.push({
      id, name: payload.name || `${user}'s model`, owner: user,
      classes: payload.classes, feat_dim: weights.feat, hidden: weights.hidden,
      n_samples: payload.n_samples || 0, created: now, weights,
    });
    this._save(models);
    return { id, name: payload.name || `${user}'s model` };
  }
  async infer(modelId, features, user) {
    const m = this._load().find((x) => x.id === modelId);
    if (!m) return { error: "unknown model" };
    const probs = runHead(m.weights, features);
    const predictions = m.classes
      .map((label, i) => ({ label, prob: probs[i] }))
      .sort((a, b) => b.prob - a.prob);
    return {
      model: m.name, owner: m.owner, predictions,
      note: "The head ran locally in the browser (localStorage) — no server.",
    };
  }
}

// Pick a backend: use the server if the API responds, otherwise local
// (Pages/offline).
export async function makeBackend() {
  const onPages = location.hostname.endsWith("github.io") || location.protocol === "file:";
  if (!onPages) {
    try {
      const r = await fetch(api("api/users"), { method: "GET" });
      if (r.ok) return new ServerBackend();
    } catch { /* no server */ }
  }
  return new LocalBackend();
}
