// WebGL2 GPU backbone. Replicates EXACTLY the same conv pipeline as Rust/WASM
// (`extract_features`), but on the GPU via fragment shaders. Uses the identical,
// deterministic weights that WASM exports (`backbone_weights_json`), so GPU and
// CPU features are compatible.
//
// Feature maps are stored as "tiled" RGBA32F textures: channel c goes into tile
// (c % tilesX, c / tilesX). The value lives in .r.
//
// Requires WebGL2 + EXT_color_buffer_float (to render to float textures). If
// anything is missing, tryCreate(...) returns null and we fall back to WASM.

const VERT = `#version 300 es
void main(){
  vec2 v[3] = vec2[3](vec2(-1.,-1.), vec2(3.,-1.), vec2(-1.,3.));
  gl_Position = vec4(v[gl_VertexID], 0., 1.);
}`;

const CONV_FRAG = `#version 300 es
precision highp float; precision highp int;
uniform sampler2D uInput;   // tiled input feature-map
uniform sampler2D uW;       // weights: width=inC*K*K, height=outC
uniform float uBias[32];
uniform int uOutC, uOutTilesX, uOutW, uOutH;
uniform int uInTilesX, uInW, uInH, uInC;
uniform int uK, uDoRelu;
out vec4 frag;
void main(){
  ivec2 p = ivec2(gl_FragCoord.xy);
  int tileX = p.x / uOutW; int ox = p.x - tileX*uOutW;
  int tileY = p.y / uOutH; int oy = p.y - tileY*uOutH;
  int oc = tileY*uOutTilesX + tileX;
  if(oc >= uOutC){ frag = vec4(0.); return; }
  float acc = uBias[oc];
  for(int ic=0; ic<uInC; ic++){
    int itx = ic % uInTilesX; int ity = ic / uInTilesX;
    for(int ky=0; ky<uK; ky++){
      for(int kx=0; kx<uK; kx++){
        int px = itx*uInW + (ox+kx);
        int py = ity*uInH + (oy+ky);
        float xv = texelFetch(uInput, ivec2(px,py), 0).r;
        float wv = texelFetch(uW, ivec2(ic*uK*uK + ky*uK + kx, oc), 0).r;
        acc += xv*wv;
      }
    }
  }
  if(uDoRelu == 1) acc = max(acc, 0.0);
  frag = vec4(acc, 0., 0., 1.);
}`;

const POOL_FRAG = `#version 300 es
precision highp float; precision highp int;
uniform sampler2D uInput;
uniform int uC, uOutTilesX, uOutW, uOutH;
uniform int uInTilesX, uInW, uInH;
out vec4 frag;
void main(){
  ivec2 p = ivec2(gl_FragCoord.xy);
  int tx = p.x / uOutW; int ox = p.x - tx*uOutW;
  int ty = p.y / uOutH; int oy = p.y - ty*uOutH;
  int c = ty*uOutTilesX + tx;
  if(c >= uC){ frag = vec4(0.); return; }
  int itx = c % uInTilesX; int ity = c / uInTilesX;
  float m = -3.4e38;
  for(int dy=0; dy<2; dy++){
    for(int dx=0; dx<2; dx++){
      int px = itx*uInW + (ox*2+dx);
      int py = ity*uInH + (oy*2+dy);
      m = max(m, texelFetch(uInput, ivec2(px,py), 0).r);
    }
  }
  frag = vec4(m, 0., 0., 1.);
}`;

const GAP_FRAG = `#version 300 es
precision highp float; precision highp int;
uniform sampler2D uInput;
uniform int uC, uInTilesX, uInW, uInH;
out vec4 frag;
void main(){
  int c = int(gl_FragCoord.x);   // output: width=C, height=1
  if(c >= uC){ frag = vec4(0.); return; }
  int itx = c % uInTilesX; int ity = c / uInTilesX;
  float s = 0.0;
  for(int y=0; y<uInH; y++){
    for(int x=0; x<uInW; x++){
      s += texelFetch(uInput, ivec2(itx*uInW + x, ity*uInH + y), 0).r;
    }
  }
  frag = vec4(s / float(uInW*uInH), 0., 0., 1.);
}`;

function tilesFor(c) {
  const tx = Math.ceil(Math.sqrt(c));
  const ty = Math.ceil(c / tx);
  return { tx, ty };
}

function compile(gl, type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    throw new Error("shader: " + gl.getShaderInfoLog(sh));
  }
  return sh;
}

function program(gl, frag) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, VERT));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, frag));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    throw new Error("link: " + gl.getProgramInfoLog(p));
  }
  return p;
}

export class GpuBackbone {
  static tryCreate(weights) {
    try {
      const canvas = document.createElement("canvas");
      const gl = canvas.getContext("webgl2", { antialias: false });
      if (!gl) return null;
      if (!gl.getExtension("EXT_color_buffer_float")) return null;
      return new GpuBackbone(gl, weights);
    } catch (e) {
      console.warn("GPU backbone unavailable:", e);
      return null;
    }
  }

  constructor(gl, w) {
    this.gl = gl;
    this.IN = w.in_size; // 48
    this.K = w.k;
    this.inC = w.in_c; // 3
    this.C1 = w.c1;
    this.C2 = w.c2;
    this.O1 = this.IN - this.K + 1; // 46
    this.P1 = (this.O1 / 2) | 0; // 23
    this.O2 = this.P1 - this.K + 1; // 21

    this.progConv = program(gl, CONV_FRAG);
    this.progPool = program(gl, POOL_FRAG);
    this.progGap = program(gl, GAP_FRAG);

    // Weight textures (sample-only, R32F)
    this.w1Tex = this._weightTex(this.inC * this.K * this.K, this.C1, w.c1_w);
    this.w2Tex = this._weightTex(this.C1 * this.K * this.K, this.C2, w.c2_w);
    this.b1 = this._pad32(w.c1_b);
    this.b2 = this._pad32(w.c2_b);

    // Tilings
    this.inTiles = tilesFor(this.inC);
    this.t1 = tilesFor(this.C1);
    this.t2 = tilesFor(this.C2);

    // Render targets (RGBA32F) + input texture (R32F)
    const inW = this.inTiles.tx * this.IN, inH = this.inTiles.ty * this.IN;
    this.inputTex = this._dataTex(inW, inH, null, true);
    this.inW = inW; this.inH = inH;

    this.a1 = this._target(this.t1.tx * this.O1, this.t1.ty * this.O1);
    this.p1 = this._target(this.t1.tx * this.P1, this.t1.ty * this.P1);
    this.c2 = this._target(this.t2.tx * this.O2, this.t2.ty * this.O2);
    this.gap = this._target(this.C2, 1);

    this.vao = gl.createVertexArray();
  }

  _pad32(arr) {
    const out = new Float32Array(32);
    out.set(arr.slice(0, 32));
    return out;
  }

  _weightTex(w, h, data) {
    const gl = this.gl;
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, w, h, 0, gl.RED, gl.FLOAT,
      new Float32Array(data));
    this._nearest();
    return t;
  }

  _dataTex(w, h, data, keep) {
    const gl = this.gl;
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, w, h, 0, gl.RED, gl.FLOAT, data);
    this._nearest();
    return t;
  }

  _target(w, h) {
    const gl = this.gl;
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, w, h, 0, gl.RGBA, gl.FLOAT, null);
    this._nearest();
    const fb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error("float framebuffer incomplete");
    }
    return { tex, fb, w, h };
  }

  _nearest() {
    const gl = this.gl;
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  _u(prog, name) {
    return this.gl.getUniformLocation(prog, name);
  }

  _draw(target) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, target.fb);
    gl.viewport(0, 0, target.w, target.h);
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  // rgba: Uint8 of length IN*IN*4 (already scaled to IN×IN by the caller)
  extract(rgba) {
    const gl = this.gl;
    const IN = this.IN, inC = this.inC, tx = this.inTiles.tx;
    const data = new Float32Array(this.inW * this.inH);
    for (let c = 0; c < inC; c++) {
      const ox = (c % tx) * IN, oy = ((c / tx) | 0) * IN;
      for (let y = 0; y < IN; y++) {
        for (let x = 0; x < IN; x++) {
          const v = rgba[(y * IN + x) * 4 + c] / 255 - 0.5;
          data[(oy + y) * this.inW + (ox + x)] = v;
        }
      }
    }
    gl.bindTexture(gl.TEXTURE_2D, this.inputTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, this.inW, this.inH, 0, gl.RED, gl.FLOAT, data);

    // ---- conv1 + relu ----
    this._conv(this.progConv, this.inputTex, this.w1Tex, this.b1,
      this.C1, this.t1, this.O1, this.O1,
      this.inTiles, IN, IN, inC, 1, this.a1);
    // ---- maxpool ----
    gl.useProgram(this.progPool);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.a1.tex);
    gl.uniform1i(this._u(this.progPool, "uInput"), 0);
    gl.uniform1i(this._u(this.progPool, "uC"), this.C1);
    gl.uniform1i(this._u(this.progPool, "uOutTilesX"), this.t1.tx);
    gl.uniform1i(this._u(this.progPool, "uOutW"), this.P1);
    gl.uniform1i(this._u(this.progPool, "uOutH"), this.P1);
    gl.uniform1i(this._u(this.progPool, "uInTilesX"), this.t1.tx);
    gl.uniform1i(this._u(this.progPool, "uInW"), this.O1);
    gl.uniform1i(this._u(this.progPool, "uInH"), this.O1);
    this._draw(this.p1);
    // ---- conv2 + relu ----
    this._conv(this.progConv, this.p1.tex, this.w2Tex, this.b2,
      this.C2, this.t2, this.O2, this.O2,
      this.t1, this.P1, this.P1, this.C1, 1, this.c2);
    // ---- global average pooling ----
    gl.useProgram(this.progGap);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.c2.tex);
    gl.uniform1i(this._u(this.progGap, "uInput"), 0);
    gl.uniform1i(this._u(this.progGap, "uC"), this.C2);
    gl.uniform1i(this._u(this.progGap, "uInTilesX"), this.t2.tx);
    gl.uniform1i(this._u(this.progGap, "uInW"), this.O2);
    gl.uniform1i(this._u(this.progGap, "uInH"), this.O2);
    this._draw(this.gap);

    // ---- read out + L2-normalize ----
    const px = new Float32Array(this.C2 * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.gap.fb);
    gl.readPixels(0, 0, this.C2, 1, gl.RGBA, gl.FLOAT, px);
    const feat = new Float32Array(this.C2);
    let norm = 0;
    for (let i = 0; i < this.C2; i++) { feat[i] = px[i * 4]; norm += feat[i] * feat[i]; }
    norm = Math.sqrt(norm) || 1e-8;
    for (let i = 0; i < this.C2; i++) feat[i] /= norm;
    return feat;
  }

  _conv(prog, inputTex, wTex, bias, outC, outT, outW, outH, inT, inW, inH, inC, doRelu, target) {
    const gl = this.gl;
    gl.useProgram(prog);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, inputTex);
    gl.uniform1i(this._u(prog, "uInput"), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, wTex);
    gl.uniform1i(this._u(prog, "uW"), 1);
    gl.uniform1fv(this._u(prog, "uBias[0]"), bias);
    gl.uniform1i(this._u(prog, "uOutC"), outC);
    gl.uniform1i(this._u(prog, "uOutTilesX"), outT.tx);
    gl.uniform1i(this._u(prog, "uOutW"), outW);
    gl.uniform1i(this._u(prog, "uOutH"), outH);
    gl.uniform1i(this._u(prog, "uInTilesX"), inT.tx);
    gl.uniform1i(this._u(prog, "uInW"), inW);
    gl.uniform1i(this._u(prog, "uInH"), inH);
    gl.uniform1i(this._u(prog, "uInC"), inC);
    gl.uniform1i(this._u(prog, "uK"), this.K);
    gl.uniform1i(this._u(prog, "uDoRelu"), doRelu);
    this._draw(target);
  }
}
