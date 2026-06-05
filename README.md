# SplitAI

Lokalt Django-prosjekt som demonstrerer en **delt «trakt»-modell** der de tidlige
lagene (backbone) kjører i nettleseren via **Rust → WASM** (med **WebGL2-GPU** når
det er støttet), og de siste lagene (hodet) trenes lokalt og kjøres delt på serveren
— uten at de trente vektene avsløres for andre brukere.

```
  Nettleser (bruker A)                         Server (Django + numpy)
  ┌────────────────────────────┐               ┌──────────────────────────┐
  │ bilde → backbone (WASM/GPU) │  features     │   hode (siste lag)       │
  │   faste, delte vekter       │ ───────────▶  │   hemmelige, trente      │
  │   → 32-d feature-vektor     │               │   vekter → svar          │
  └────────────────────────────┘               └──────────────────────────┘
        ▲ trener hodet lokalt                         ▲ lagrer hodet
        └── eksporterer hode-vekter ──────────────────┘
```

## Idé

* **Backbone (tidlige lag):** conv-lag med **faste, deterministiske vekter** som er
  like for alle klienter. Kjøres i nettleseren. Gir en 32-dim feature-vektor.
  Tung del av «trakten», men frosset.
* **Hode (siste lag):** et lite MLP. Dette **trenes** via transfer learning i
  nettleseren (på frosne features) og er det «verdifulle/hemmelige».
* **Deling uten å avsløre vekter:** Hode-vektene lagres kun på serveren. Andre
  brukere kjører backbone lokalt på *sine* bilder og sender **kun feature-vektoren**
  til serveren, som kjører hodet og returnerer svaret. Vektene forlater aldri serveren.
* **GPU via WebGL2:** Backbone-en kjøres på GPU når nettleseren støtter WebGL2 +
  `EXT_color_buffer_float`, ellers faller den tilbake til CPU/WASM. En innebygd
  selvtest sjekker at GPU- og CPU-features matcher (CPU er fasiten alle deler); en
  klient bruker bare GPU hvis dens egen test matcher.

> **Forbehold:** Dette er en kompakt YOLO-*inspirert* CNN (conv-backbone + MLP-hode),
> ikke full YOLO. Poenget er å vise split/trakt-pipelinen ende-til-ende. Arkitekturen
> i `wasm/src/lib.rs` kan byttes ut med en større/ekte backbone senere — server- og
> frontend-flyten er uendret så lenge feature-dimensjonen følger med.

## Krav

* Python 3.12 (venv ligger i `venv/`)
* Rust + `wasm-pack` (kun for å *bygge* WASM på nytt — ferdig bygd ligger i `static/wasm/`)

## Kom i gang

Vanlig HTTP (alt unntatt webkamera):

```bash
./run.sh                      # migrerer + starter http://127.0.0.1:8000/
```

**Med webkamera (HTTPS, virker på telefon over LAN):**

```bash
./run_https.sh                # → https://<din-LAN-IP>:8443/  (selvsignert cert)
```

Webkamera (`getUserMedia`) krever «secure context», så live stream funker bare på
`https://` eller `localhost` — derfor HTTPS-varianten. Godta sertifikat-advarselen
én gang per enhet. (`run_https.sh [port]` for annen port.)

eller manuelt:

```bash
python3.12 -m venv venv
./venv/bin/pip install -r requirements.txt
./build_wasm.sh               # bygger Rust → static/wasm/  (valgfritt, alt bygd)
./venv/bin/python manage.py migrate
./venv/bin/python manage.py runserver
```

## Kjøre kun i nettleseren (GitHub Pages)

Appen kjører **også helt uten server**. `static/js/backend.js` velger automatisk:

* **ServerBackend** når Django-API-et svarer (hovedmotoren — modeller i sqlite).
* **LocalBackend** ellers (f.eks. GitHub Pages) — modeller lagres i `localStorage`
  og hodet kjøres lokalt i nettleseren (`runHead` speiler `core/head_runtime.py`).
  Hode-vektene er bare noen få kB, så de får lett plass lokalt.

Alle stier er relative, så det samme `index.html` virker både under Django (`/`)
og under Pages (`/splitai/`). GitHub Pages gir HTTPS, så **webkamera virker uten
egen sertifikat-oppsett** der.

Aktiver Pages: repo → Settings → Pages → «Deploy from a branch» → `main` / `/ (root)`.
Deretter: `https://punnerud.github.io/splitai/`. (Statuslinja i appen viser om
lagring er «server» eller «localStorage».) Merk: i Pages-modus deles ikke modeller
mellom enheter — `localStorage` er per nettleser. For deling på tvers: kjør Django.

## Default-modell: YOLO (live deteksjon → annotér → retren)

Seksjon 0 i appen kjører en **forhåndstrent YOLOv10n** (COCO, 80 klasser inkl.
person, kopp, flaske …) i nettleseren via **onnxruntime-web** (WebGPU når støttet,
ellers WASM — alt vendret lokalt i `static/vendor/ort/` + `static/models/`).

Arbeidsflyt:
1. *Start kamera* (HTTPS) eller *last opp bilde* → YOLO tegner bokser live.
2. *Ta bilde* → bildet fryses og YOLO foreslår bokser.
3. **Annotér:** dra for å tegne nye bokser, dra/klikk for å flytte/velge, rett
   etiketter eller slett i lista.
4. *Legg bokser i treningssett* → hvert utsnitt klippes ut, kjøres gjennom
   backbone-en (WASM/GPU) til en feature-vektor, og havner i steg 2.
5. *Tren hodet* (steg 2) → *Lagre hode til server* → del/kjør som vanlig.

> **«Retren» = hodet, ikke YOLO.** YOLO holdes frosset som boks-detektor. Det som
> faktisk retrenes er det lette MLP-hodet (transfer learning på de annoterte
> utsnittene). Ekte ende-til-ende YOLO-trening i nettleseren er ikke gjort — det
> krever full deteksjons-backprop og er urealistisk på klientsiden.

## Teste flere brukere

Ingen innlogging. Hver nettleser velger et navn (lagres i `localStorage`).
Åpne appen i **to ulike nettlesere eller profiler**:

1. **Bruker A (f.eks. «alice»):** velg navn → last opp merkede bilder → *Tren hodet*
   → *Lagre hode til server*.
2. **Bruker B (f.eks. «bob»):** velg navn → *Oppdater modelliste* → velg alice sin
   modell → last opp et bilde → *Kjør*. Backbone kjøres lokalt hos B; bare
   feature-vektoren sendes, og alice sine hode-vekter forblir på serveren.

## Filer

| Sti | Hva |
|---|---|
| `wasm/src/lib.rs` | Rust: backbone (`extract_features`) + MLP-hode (`MlpHead`) |
| `static/js/gpu_backbone.js` | WebGL2-GPU-utgave av backbone (identiske vekter) |
| `static/js/yolo.js` | YOLOv10n via onnxruntime-web (pre/post-prosessering) |
| `static/js/detect_ui.js` | Webkamera, live deteksjon, annotering, → treningssett |
| `static/js/app.js` | Frontend: trening, lagring, delt inferens, selvtest |
| `static/models/`, `static/vendor/ort/` | YOLO-modell + onnxruntime-web (vendret) |
| `core/head_runtime.py` | numpy-kjøring av hodet på serveren (matcher Rust) |
| `core/views.py` | API: `/api/users`, `/api/models`, `/api/infer` |
| `run_https.sh` | HTTPS-server (selvsignert cert) for webkamera over LAN |

## Tester

```bash
cd wasm && cargo test            # backbone normalisering + at hodet konvergerer
```

Numpy-hodet (server) matcher Rust-`predict` til ~1e-9 på samme vekter/features.
