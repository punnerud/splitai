//! SplitAI WASM-kjerne.
//!
//! Inneholder to deler av "trakten":
//!   * `Backbone` — faste, deterministiske conv-lag (de tidlige lagene). Kjorer i
//!     nettleseren og produserer en 32-dimensjonal feature-vektor per bilde.
//!     Vektene er identiske for alle klienter (seedet PRNG), slik at features fra
//!     ulike brukere/nettlesere er kompatible. WebGL-stien i JS bruker NOYAKTIG
//!     de samme vektene (eksportert via `backbone_weights_json`).
//!   * `MlpHead` — de siste lagene (et lite MLP). Dette er det som trenes via
//!     transfer learning, og som holdes hemmelig pa serveren under delt inferens.

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

// ---- Arkitektur-konstanter -------------------------------------------------
const IN: usize = 48; // backbone-input er IN x IN RGB
const IN_C: usize = 3;
const C1: usize = 8; // antall filtre i conv1
const C2: usize = 32; // antall filtre i conv2 == feature-dimensjon (etter GAP)
const K: usize = 3; // kernel-storrelse (3x3, "valid")

const O1: usize = IN - K + 1; // 46
const P1: usize = O1 / 2; // 23 (etter 2x2 maxpool)
const O2: usize = P1 - K + 1; // 21

/// Feature-dimensjonen backbone-en gir ut.
#[wasm_bindgen]
pub fn feat_dim() -> usize {
    C2
}

/// Input-storrelsen backbone-en forventer (kvadratisk).
#[wasm_bindgen]
pub fn input_size() -> usize {
    IN
}

// ---- Deterministisk PRNG (xorshift64) --------------------------------------
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(if seed == 0 { 0x9E3779B97F4A7C15 } else { seed })
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    /// Uniform i [-1, 1).
    fn signed(&mut self) -> f32 {
        let u = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64; // [0,1)
        (u * 2.0 - 1.0) as f32
    }
}

// ---- Backbone-vekter (faste, delte) ----------------------------------------
#[derive(Serialize, Deserialize, Clone)]
struct BackboneWeights {
    in_size: usize,
    in_c: usize,
    c1: usize,
    c2: usize,
    k: usize,
    c1_w: Vec<f32>, // [C1][IN_C][K][K]
    c1_b: Vec<f32>, // [C1]
    c2_w: Vec<f32>, // [C2][C1][K][K]
    c2_b: Vec<f32>, // [C2]
}

fn make_backbone() -> BackboneWeights {
    // Egne seeds for hvert lag. He-aktig skalering; eksakt skala spiller liten
    // rolle siden vi L2-normaliserer feature-vektoren til slutt.
    let mut r1 = Rng::new(0xA11CE5);
    let fan1 = (IN_C * K * K) as f32;
    let s1 = (2.0f32 / fan1).sqrt();
    let mut c1_w = vec![0.0f32; C1 * IN_C * K * K];
    for w in c1_w.iter_mut() {
        *w = r1.signed() * s1;
    }
    let c1_b = vec![0.0f32; C1];

    let mut r2 = Rng::new(0xB0B5EED);
    let fan2 = (C1 * K * K) as f32;
    let s2 = (2.0f32 / fan2).sqrt();
    let mut c2_w = vec![0.0f32; C2 * C1 * K * K];
    for w in c2_w.iter_mut() {
        *w = r2.signed() * s2;
    }
    let c2_b = vec![0.0f32; C2];

    BackboneWeights {
        in_size: IN,
        in_c: IN_C,
        c1: C1,
        c2: C2,
        k: K,
        c1_w,
        c1_b,
        c2_w,
        c2_b,
    }
}

/// Eksporter backbone-vektene som JSON. WebGL-stien laster disse inn i teksturer
/// slik at GPU- og CPU-stien bruker identiske vekter.
#[wasm_bindgen]
pub fn backbone_weights_json() -> String {
    serde_json::to_string(&make_backbone()).unwrap()
}

#[inline]
fn relu(x: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

/// Skaler vilkarlig RGBA-buffer (lengde w*h*4) ned til IN x IN x 3, normalisert
/// til [-0.5, 0.5]. Nearest-neighbor — JS skalerer normalt allerede via canvas.
fn to_input(rgba: &[u8], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; IN_C * IN * IN];
    for oy in 0..IN {
        let sy = oy * h / IN;
        for ox in 0..IN {
            let sx = ox * w / IN;
            let src = (sy * w + sx) * 4;
            for c in 0..IN_C {
                let v = rgba[src + c] as f32 / 255.0 - 0.5;
                out[c * IN * IN + oy * IN + ox] = v;
            }
        }
    }
    out
}

/// Kjor backbone pa et bilde og returner L2-normalisert feature-vektor (lengde 32).
/// `rgba` er en RGBA-buffer av storrelse `w` x `h`.
#[wasm_bindgen]
pub fn extract_features(rgba: &[u8], w: usize, h: usize) -> Vec<f32> {
    let bb = make_backbone();
    let x = to_input(rgba, w, h);

    // conv1 + relu  ->  [C1][O1][O1]
    let mut a1 = vec![0.0f32; C1 * O1 * O1];
    for oc in 0..C1 {
        for oy in 0..O1 {
            for ox in 0..O1 {
                let mut acc = bb.c1_b[oc];
                for ic in 0..IN_C {
                    for ky in 0..K {
                        for kx in 0..K {
                            let xv = x[ic * IN * IN + (oy + ky) * IN + (ox + kx)];
                            let wv = bb.c1_w[((oc * IN_C + ic) * K + ky) * K + kx];
                            acc += xv * wv;
                        }
                    }
                }
                a1[oc * O1 * O1 + oy * O1 + ox] = relu(acc);
            }
        }
    }

    // maxpool 2x2 stride 2  ->  [C1][P1][P1]
    let mut p1 = vec![0.0f32; C1 * P1 * P1];
    for c in 0..C1 {
        for py in 0..P1 {
            for px in 0..P1 {
                let mut m = f32::MIN;
                for dy in 0..2 {
                    for dx in 0..2 {
                        let v = a1[c * O1 * O1 + (py * 2 + dy) * O1 + (px * 2 + dx)];
                        if v > m {
                            m = v;
                        }
                    }
                }
                p1[c * P1 * P1 + py * P1 + px] = m;
            }
        }
    }

    // conv2 + relu  ->  [C2][O2][O2], deretter global average pooling -> [C2]
    let mut feat = vec![0.0f32; C2];
    for oc in 0..C2 {
        let mut sum = 0.0f32;
        for oy in 0..O2 {
            for ox in 0..O2 {
                let mut acc = bb.c2_b[oc];
                for ic in 0..C1 {
                    for ky in 0..K {
                        for kx in 0..K {
                            let pv = p1[ic * P1 * P1 + (oy + ky) * P1 + (ox + kx)];
                            let wv = bb.c2_w[((oc * C1 + ic) * K + ky) * K + kx];
                            acc += pv * wv;
                        }
                    }
                }
                sum += relu(acc);
            }
        }
        feat[oc] = sum / (O2 * O2) as f32;
    }

    // L2-normaliser
    let mut norm = 0.0f32;
    for v in &feat {
        norm += v * v;
    }
    norm = norm.sqrt().max(1e-8);
    for v in feat.iter_mut() {
        *v /= norm;
    }
    feat
}

// ---- MLP-hode (de "siste lagene" som trenes) -------------------------------
#[derive(Serialize, Deserialize)]
struct HeadWeights {
    feat: usize,
    hidden: usize,
    classes: usize,
    w1: Vec<f32>, // [hidden][feat]
    b1: Vec<f32>, // [hidden]
    w2: Vec<f32>, // [classes][hidden]
    b2: Vec<f32>, // [classes]
}

#[wasm_bindgen]
pub struct MlpHead {
    feat: usize,
    hidden: usize,
    classes: usize,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    xs: Vec<f32>,    // treningsfeatures, flatt [n][feat]
    ys: Vec<usize>,  // treningslabels, [n]
}

#[wasm_bindgen]
impl MlpHead {
    /// Nytt hode med tilfeldig (seedet) init. `hidden` er skjult lagstorrelse.
    #[wasm_bindgen(constructor)]
    pub fn new(classes: usize, hidden: usize) -> MlpHead {
        let feat = C2;
        let mut r = Rng::new(0xC0FFEE ^ (classes as u64) << 8 ^ (hidden as u64));
        let s1 = (2.0f32 / feat as f32).sqrt();
        let mut w1 = vec![0.0f32; hidden * feat];
        for w in w1.iter_mut() {
            *w = r.signed() * s1;
        }
        let s2 = (2.0f32 / hidden as f32).sqrt();
        let mut w2 = vec![0.0f32; classes * hidden];
        for w in w2.iter_mut() {
            *w = r.signed() * s2;
        }
        MlpHead {
            feat,
            hidden,
            classes,
            w1,
            b1: vec![0.0; hidden],
            w2,
            b2: vec![0.0; classes],
            xs: Vec::new(),
            ys: Vec::new(),
        }
    }

    /// Legg til et treningseksempel (feature-vektor + klasseindeks).
    pub fn add_sample(&mut self, feat: &[f32], label: usize) {
        assert_eq!(feat.len(), self.feat);
        self.xs.extend_from_slice(feat);
        self.ys.push(label);
    }

    pub fn num_samples(&self) -> usize {
        self.ys.len()
    }

    fn forward(&self, x: &[f32], a1: &mut [f32], out: &mut [f32]) {
        for h in 0..self.hidden {
            let mut acc = self.b1[h];
            let row = h * self.feat;
            for f in 0..self.feat {
                acc += self.w1[row + f] * x[f];
            }
            a1[h] = relu(acc);
        }
        for c in 0..self.classes {
            let mut acc = self.b2[c];
            let row = c * self.hidden;
            for h in 0..self.hidden {
                acc += self.w2[row + h] * a1[h];
            }
            out[c] = acc;
        }
        // softmax in-place
        let mut mx = f32::MIN;
        for &v in out.iter() {
            if v > mx {
                mx = v;
            }
        }
        let mut s = 0.0f32;
        for v in out.iter_mut() {
            *v = (*v - mx).exp();
            s += *v;
        }
        for v in out.iter_mut() {
            *v /= s;
        }
    }

    /// Kjor en treningsepoke (full-batch SGD). Returnerer snitt-cross-entropy.
    pub fn train_epoch(&mut self, lr: f32) -> f32 {
        let n = self.ys.len();
        if n == 0 {
            return 0.0;
        }
        let (feat, hidden, classes) = (self.feat, self.hidden, self.classes);
        let mut gw1 = vec![0.0f32; hidden * feat];
        let mut gb1 = vec![0.0f32; hidden];
        let mut gw2 = vec![0.0f32; classes * hidden];
        let mut gb2 = vec![0.0f32; classes];
        let mut a1 = vec![0.0f32; hidden];
        let mut out = vec![0.0f32; classes];
        let mut loss = 0.0f32;

        for i in 0..n {
            let x = &self.xs[i * feat..(i + 1) * feat];
            self.forward(x, &mut a1, &mut out);
            let y = self.ys[i];
            loss -= out[y].max(1e-12).ln();

            // dL/dz2 = softmax - onehot
            let mut dz2 = out.clone();
            dz2[y] -= 1.0;

            // grad hode-lag 2 + propagering til a1
            let mut da1 = vec![0.0f32; hidden];
            for c in 0..classes {
                let g = dz2[c];
                gb2[c] += g;
                let row = c * hidden;
                for h in 0..hidden {
                    gw2[row + h] += g * a1[h];
                    da1[h] += g * self.w2[row + h];
                }
            }
            // gjennom relu
            for h in 0..hidden {
                let dz1 = if a1[h] > 0.0 { da1[h] } else { 0.0 };
                gb1[h] += dz1;
                let row = h * feat;
                for f in 0..feat {
                    gw1[row + f] += dz1 * x[f];
                }
            }
        }

        let scale = lr / n as f32;
        for j in 0..gw1.len() {
            self.w1[j] -= scale * gw1[j];
        }
        for j in 0..gb1.len() {
            self.b1[j] -= scale * gb1[j];
        }
        for j in 0..gw2.len() {
            self.w2[j] -= scale * gw2[j];
        }
        for j in 0..gb2.len() {
            self.b2[j] -= scale * gb2[j];
        }
        loss / n as f32
    }

    /// Sannsynlighetsfordeling over klasser for en enkelt feature-vektor.
    pub fn predict(&self, feat: &[f32]) -> Vec<f32> {
        let mut a1 = vec![0.0f32; self.hidden];
        let mut out = vec![0.0f32; self.classes];
        self.forward(feat, &mut a1, &mut out);
        out
    }

    /// Eksporter hode-vektene som JSON (det som lagres pa serveren).
    pub fn export_json(&self) -> String {
        let hw = HeadWeights {
            feat: self.feat,
            hidden: self.hidden,
            classes: self.classes,
            w1: self.w1.clone(),
            b1: self.b1.clone(),
            w2: self.w2.clone(),
            b2: self.b2.clone(),
        };
        serde_json::to_string(&hw).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn features_are_normalized() {
        let rgba = vec![123u8; IN * IN * 4];
        let f = extract_features(&rgba, IN, IN);
        assert_eq!(f.len(), C2);
        let n: f32 = f.iter().map(|v| v * v).sum();
        assert!((n - 1.0).abs() < 1e-4, "norm = {n}");
    }

    #[test]
    fn training_separates_two_classes() {
        // To lineaert separerbare "features" (lengde 32).
        let mut a = vec![0.0f32; C2];
        a[0] = 1.0;
        let mut b = vec![0.0f32; C2];
        b[1] = 1.0;
        let mut head = MlpHead::new(2, 16);
        for _ in 0..5 {
            head.add_sample(&a, 0);
            head.add_sample(&b, 1);
        }
        let mut loss = 9.9;
        for _ in 0..400 {
            loss = head.train_epoch(0.5);
        }
        assert!(loss < 0.05, "tap konvergerte ikke: {loss}");
        let pa = head.predict(&a);
        let pb = head.predict(&b);
        assert!(pa[0] > 0.9 && pb[1] > 0.9, "pa={pa:?} pb={pb:?}");

        // Eksport for kryss-sjekk mot numpy-serveren.
        std::fs::write("/tmp/splitai_head.json", head.export_json()).unwrap();
        std::fs::write(
            "/tmp/splitai_feat.json",
            serde_json::to_string(&a).unwrap(),
        )
        .unwrap();
        let pred: Vec<f32> = head.predict(&a).into();
        std::fs::write("/tmp/splitai_pred.json", serde_json::to_string(&pred).unwrap())
            .unwrap();
    }
}
