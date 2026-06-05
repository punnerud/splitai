/* tslint:disable */
/* eslint-disable */

export class MlpHead {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Legg til et treningseksempel (feature-vektor + klasseindeks).
     */
    add_sample(feat: Float32Array, label: number): void;
    /**
     * Eksporter hode-vektene som JSON (det som lagres pa serveren).
     */
    export_json(): string;
    /**
     * Nytt hode med tilfeldig (seedet) init. `hidden` er skjult lagstorrelse.
     */
    constructor(classes: number, hidden: number);
    num_samples(): number;
    /**
     * Sannsynlighetsfordeling over klasser for en enkelt feature-vektor.
     */
    predict(feat: Float32Array): Float32Array;
    /**
     * Kjor en treningsepoke (full-batch SGD). Returnerer snitt-cross-entropy.
     */
    train_epoch(lr: number): number;
}

/**
 * Eksporter backbone-vektene som JSON. WebGL-stien laster disse inn i teksturer
 * slik at GPU- og CPU-stien bruker identiske vekter.
 */
export function backbone_weights_json(): string;

/**
 * Kjor backbone pa et bilde og returner L2-normalisert feature-vektor (lengde 32).
 * `rgba` er en RGBA-buffer av storrelse `w` x `h`.
 */
export function extract_features(rgba: Uint8Array, w: number, h: number): Float32Array;

/**
 * Feature-dimensjonen backbone-en gir ut.
 */
export function feat_dim(): number;

/**
 * Input-storrelsen backbone-en forventer (kvadratisk).
 */
export function input_size(): number;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_mlphead_free: (a: number, b: number) => void;
    readonly backbone_weights_json: () => [number, number];
    readonly extract_features: (a: number, b: number, c: number, d: number) => [number, number];
    readonly feat_dim: () => number;
    readonly input_size: () => number;
    readonly mlphead_add_sample: (a: number, b: number, c: number, d: number) => void;
    readonly mlphead_export_json: (a: number) => [number, number];
    readonly mlphead_new: (a: number, b: number) => number;
    readonly mlphead_num_samples: (a: number) => number;
    readonly mlphead_predict: (a: number, b: number, c: number) => [number, number];
    readonly mlphead_train_epoch: (a: number, b: number) => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
