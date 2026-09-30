// ============================================================
// 身份证 A4 拼版打印 —— 深度学习四角检测 + 透视拉正 + 方向校正
// 纯前端、离线运行：onnxruntime-web 与两个 ONNX 模型均内联为本地脚本，
// 仅在首次识别时懒加载，照片不离开本机。
// ============================================================

// ===== A4 / 卡片常量（300dpi）=====
const A4_W = 2480, A4_H = 3508;
const CARD_RATIO = 85.6 / 54;                       // 身份证真实长宽比
const CARD_W = Math.round(A4_W * 85.6 / 210);       // 卡片在 A4 上的宽
const CARD_H = Math.round(CARD_W / CARD_RATIO);
const CORNER_R = Math.round(300 * 3 / 25.4);        // 3mm 圆角
const GAP = Math.round(A4_H * 60 / 297);            // 两面间距

const DET_SIZE = 256, GRID = 128, DET_CONF = 0.05, WARP_MAX = 1500;
const ORI_SIZE = 224;
const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];

// ===== 状态 =====
const state = {
  front: { img: null, canvas: null, rawCanvas: null, rot: 0 },
  back:  { img: null, canvas: null, rawCanvas: null, rot: 0 },
  brightness: 50,
  sharpen: 0,
};

// ============================================================
// 识别引擎：onnxruntime-web（内联 wasm）+ 两个内联模型，懒加载
// ============================================================
let enginePromise = null;
const loadedScripts = new Set();

function loadScript(src) {
  if (loadedScripts.has(src)) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const el = document.createElement('script');
    el.src = src; el.async = true;
    el.onload = () => { loadedScripts.add(src); resolve(); };
    el.onerror = () => reject(new Error('加载失败: ' + src));
    document.head.appendChild(el);
  });
}
async function b64ToBytes(b64) {
  // 优先用浏览器原生 data:URI 解码（C++ 实现，比 atob+逐字节循环快数倍且不卡主线程）
  try {
    const res = await fetch('data:application/octet-stream;base64,' + b64);
    return new Uint8Array(await res.arrayBuffer());
  } catch (e) {
    const bin = atob(b64), len = bin.length, a = new Uint8Array(len);
    for (let i = 0; i < len; i++) a[i] = bin.charCodeAt(i);
    return a;
  }
}

function setEngineStatus(s, msg) {
  const el = document.getElementById('engine-status');
  if (!el) return;
  if (s === 'loading') {
    el.style.display = 'flex';
    el.className = 'engine-status loading';
    el.innerHTML = '<span class="dot"></span>' + (msg || '识别引擎加载中（首次约几秒）…');
  } else if (s === 'ready') {
    el.style.display = 'none';
  } else {
    el.style.display = 'flex';
    el.className = 'engine-status error';
    el.innerHTML = '<span class="dot"></span>' + (msg || '识别引擎加载失败，请刷新重试');
  }
}

let engineReady = false;
async function ensureEngine(silent) {
  if (enginePromise) {
    if (!silent && !engineReady) setEngineStatus('loading');
    return enginePromise;
  }
  enginePromise = (async () => {
    if (!silent) setEngineStatus('loading');
    await loadScript('js/ort/ort.umd.js');
    const ort = window.ort;
    if (!ort || !ort.InferenceSession) throw new Error('onnxruntime 未就绪');
    // 三个内联大文件（wasm + 两个模型）并行加载，避免串行等待
    await Promise.all([
      loadScript('js/ort/ort-wasm-bin.js'),
      loadScript('assets/models/model-docaligner.js'),
      loadScript('assets/models/model-docori.js'),
    ]);
    // file:// 下无法 fetch 本地 wasm，用 data:URI 经 wasmPaths 映射（data: 协议允许）；单线程 SIMD
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;
    ort.env.wasm.proxy = false;
    ort.env.debug = false;
    ort.env.logLevel = 'error';
    ort.env.wasm.wasmPaths = {
      'ort-wasm-simd.wasm': 'data:application/wasm;base64,' + window.__ORT_WASM_B64
    };
    await new Promise(r => setTimeout(r, 30));   // 让出主线程，先绘制加载提示
    if (!silent) setEngineStatus('loading', '正在初始化识别模型（首次约几秒）…');
    const opt = { executionProviders: ['wasm'], graphOptimizationLevel: 'basic' };
    // 两个模型并行初始化（wasm 内核在首次创建时一次性编译，第二个模型直接复用）
    const [sq, so] = await Promise.all([
      ort.InferenceSession.create(await b64ToBytes(window.__MODEL_DOCALIGNER_B64), opt),
      ort.InferenceSession.create(await b64ToBytes(window.__MODEL_DOCORI_B64), opt),
    ]);
    engineReady = true;
    window.__engineReadyAt = Math.round(performance.now());
    setEngineStatus('ready');
    return { ort, sq, so };
  })().catch(e => { enginePromise = null; engineReady = false; setEngineStatus('error'); throw e; });
  return enginePromise;
}

// ============================================================
// 四角检测（DocAligner LCNet，输出 TL/TR/BR/BL 热力图）
// ============================================================
function decodeHeat(t, W, H, thr) {
  const n = GRID, aw = W / n, ah = H / n, r = [];
  for (let l = 0; l < 4; l++) {
    const base = l * n * n;
    let pk = -Infinity, bi = 0;
    for (let u = 0; u < n * n; u++) { const g = t[base + u] ?? 0; if (g > pk) { pk = g; bi = u; } }
    if (pk < thr) { r.push(null); continue; }
    const gx = bi % n, gy = (bi - gx) / n;
    const xL = Math.max(0, gx - 4), xR = Math.min(n - 1, gx + 4);
    const yT = Math.max(0, gy - 4), yB = Math.min(n - 1, gy + 4);
    const wThr = 0.3 * pk;
    let sw = 0, sx = 0, sy = 0;
    for (let y = yT; y <= yB; y++) for (let x = xL; x <= xR; x++) {
      const v = t[base + y * n + x] ?? 0;
      if (v >= wThr) { sw += v; sx += v * x; sy += v * y; }
    }
    r.push({ x: (sw > 0 ? sx / sw : gx) * aw, y: (sw > 0 ? sy / sw : gy) * ah, c: pk });
  }
  return r;
}
// 缺一个角时按平行四边形补全（TL+BR = TR+BL）
function fillCorners(q) {
  let filled = 0;
  const set = (i, x, y) => { q[i] = { x, y, c: 0.3, fill: 1 }; filled++; };
  if (!q[0] && q[1] && q[2] && q[3]) set(0, q[1].x + q[3].x - q[2].x, q[1].y + q[3].y - q[2].y);
  if (!q[1] && q[0] && q[2] && q[3]) set(1, q[0].x + q[2].x - q[3].x, q[0].y + q[2].y - q[3].y);
  if (!q[2] && q[0] && q[1] && q[3]) set(2, q[1].x + q[3].x - q[0].x, q[1].y + q[3].y - q[0].y);
  if (!q[3] && q[0] && q[1] && q[2]) set(3, q[0].x + q[2].x - q[1].x, q[0].y + q[2].y - q[1].y);
  return filled;
}

async function detectQuad(eng, srcCanvas) {
  const W = srcCanvas.width, H = srcCanvas.height;
  const tc = document.createElement('canvas'); tc.width = tc.height = DET_SIZE;
  const tx = tc.getContext('2d');
  tx.drawImage(srcCanvas, 0, 0, DET_SIZE, DET_SIZE);   // 双线性拉伸，与训练一致
  const px = tx.getImageData(0, 0, DET_SIZE, DET_SIZE).data;
  const ten = new Float32Array(3 * DET_SIZE * DET_SIZE);
  const plane = DET_SIZE * DET_SIZE;
  for (let y = 0; y < DET_SIZE; y++) for (let x = 0; x < DET_SIZE; x++) {
    const i = (y * DET_SIZE + x) * 4, j = y * DET_SIZE + x;
    ten[j] = px[i] / 255; ten[plane + j] = px[i + 1] / 255; ten[2 * plane + j] = px[i + 2] / 255;
  }
  const feed = {};
  feed[eng.sq.inputNames[0]] = new eng.ort.Tensor('float32', ten, [1, 3, DET_SIZE, DET_SIZE]);
  const res = await eng.sq.run(feed);
  const hm = res[eng.sq.outputNames[0]].data;
  const q = decodeHeat(hm, W, H, DET_CONF);
  const miss = q.filter(v => !v).length;
  if (miss > 1) return null;
  fillCorners(q);
  if (q.some(v => !v)) return null;
  return { quad: q, miss };
}

// ============================================================
// 透视变换：DLT 求单应性矩阵 + 双线性反向映射（纯 JS）
// ============================================================
const pdist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
function solve8(A, b) {
  const M = A.map((r, i) => [...r, b[i]]);
  for (let s = 0; s < 8; s++) {
    let c = s, mx = Math.abs(M[s][s]);
    for (let r = s + 1; r < 8; r++) { const v = Math.abs(M[r][s]); if (v > mx) { mx = v; c = r; } }
    if (c !== s) { const tmp = M[s]; M[s] = M[c]; M[c] = tmp; }
    const pv = M[s][s];
    for (let k = s; k <= 8; k++) M[s][k] /= pv;
    for (let r = 0; r < 8; r++) {
      if (r === s) continue;
      const f = M[r][s];
      if (f !== 0) for (let k = s; k <= 8; k++) M[r][k] -= f * M[s][k];
    }
  }
  return M.map(r => r[8]);
}
function homography(src, dst) {
  const A = [], b = [];
  for (let i = 0; i < 4; i++) {
    const { x, y } = src[i], X = dst[i].x, Y = dst[i].y;
    A.push([x, y, 1, 0, 0, 0, -X * x, -X * y]); b.push(X);
    A.push([0, 0, 0, x, y, 1, -Y * x, -Y * y]); b.push(Y);
  }
  return [...solve8(A, b), 1];
}
function warpCanvas(srcCanvas, q, maxDim) {
  const W = srcCanvas.width, H = srcCanvas.height;
  let ow = Math.round(Math.max(pdist(q[0], q[1]), pdist(q[3], q[2])));
  let oh = Math.round(Math.max(pdist(q[0], q[3]), pdist(q[1], q[2])));
  const longest = Math.max(ow, oh);
  if (longest > maxDim) { const s = maxDim / longest; ow = Math.round(ow * s); oh = Math.round(oh * s); }
  const Hm = homography([{ x: 0, y: 0 }, { x: ow - 1, y: 0 }, { x: ow - 1, y: oh - 1 }, { x: 0, y: oh - 1 }], q);
  const sd = srcCanvas.getContext('2d').getImageData(0, 0, W, H).data;
  const o = new Uint8ClampedArray(ow * oh * 4).fill(255);
  for (let j = 0; j < oh; j++) for (let i = 0; i < ow; i++) {
    const den = Hm[6] * i + Hm[7] * j + Hm[8];
    const sx = (Hm[0] * i + Hm[1] * j + Hm[2]) / den;
    const sy = (Hm[3] * i + Hm[4] * j + Hm[5]) / den;
    if (sx < 0 || sy < 0 || sx > W - 1 || sy > H - 1) continue;
    const x0 = Math.floor(sx), y0 = Math.floor(sy), xx = sx - x0, yy = sy - y0;
    const x1 = Math.min(W - 1, x0 + 1), y1 = Math.min(H - 1, y0 + 1);
    const a = (y0 * W + x0) * 4, bb = (y0 * W + x1) * 4, c = (y1 * W + x0) * 4, d = (y1 * W + x1) * 4;
    const oi = (j * ow + i) * 4;
    for (let k = 0; k < 3; k++) {
      const v0 = sd[a + k] * (1 - xx) + sd[bb + k] * xx;
      const v1 = sd[c + k] * (1 - xx) + sd[d + k] * xx;
      o[oi + k] = Math.round(v0 * (1 - yy) + v1 * yy);
    }
    o[oi + 3] = 255;
  }
  const c = document.createElement('canvas'); c.width = ow; c.height = oh;
  c.getContext('2d').putImageData(new ImageData(o, ow, oh), 0, 0);
  return c;
}

// ============================================================
// 方向分类（PP-LCNet，0/90/180/270）并旋转校正
// ============================================================
async function classifyOrientation(eng, canvas) {
  const w = canvas.width, h = canvas.height;
  const s = 256 / Math.min(w, h);
  const rw = Math.max(1, Math.round(w * s)), rh = Math.max(1, Math.round(h * s));
  const rc = document.createElement('canvas'); rc.width = rw; rc.height = rh;
  rc.getContext('2d').drawImage(canvas, 0, 0, rw, rh);
  const x0 = Math.max(0, (rw - ORI_SIZE) >> 1), y0 = Math.max(0, (rh - ORI_SIZE) >> 1);
  const px = rc.getContext('2d').getImageData(x0, y0, ORI_SIZE, ORI_SIZE).data;
  const ten = new Float32Array(3 * ORI_SIZE * ORI_SIZE), plane = ORI_SIZE * ORI_SIZE;
  for (let y = 0; y < ORI_SIZE; y++) for (let x = 0; x < ORI_SIZE; x++) {
    const i = (y * ORI_SIZE + x) * 4, j = y * ORI_SIZE + x;
    ten[j] = (px[i] / 255 - MEAN[0]) / STD[0];
    ten[plane + j] = (px[i + 1] / 255 - MEAN[1]) / STD[1];
    ten[2 * plane + j] = (px[i + 2] / 255 - MEAN[2]) / STD[2];
  }
  const feed = {};
  feed[eng.so.inputNames[0]] = new eng.ort.Tensor('float32', ten, [1, 3, ORI_SIZE, ORI_SIZE]);
  const res = await eng.so.run(feed);
  const p = Array.from(res[eng.so.outputNames[0]].data);
  let bi = 0; for (let i = 1; i < 4; i++) if (p[i] > p[bi]) bi = i;
  return { angle: bi * 90, score: p[bi] };
}
function rotateCanvas(src, mode) {
  if (mode === 0) return src;
  const w = src.width, h = src.height;
  const c = document.createElement('canvas');
  const ctx = c.getContext('2d');
  if (mode === 2) {
    c.width = w; c.height = h; ctx.translate(w, h); ctx.rotate(Math.PI); ctx.drawImage(src, 0, 0);
  } else if (mode === 1) { // 顺时针 90
    c.width = h; c.height = w; ctx.translate(h, 0); ctx.rotate(Math.PI / 2); ctx.drawImage(src, 0, 0);
  } else {                 // 逆时针 90
    c.width = h; c.height = w; ctx.translate(0, w); ctx.rotate(-Math.PI / 2); ctx.drawImage(src, 0, 0);
  }
  return c;
}

async function correctIdCard(img) {
  const eng = await ensureEngine();
  const src = document.createElement('canvas');
  src.width = img.naturalWidth; src.height = img.naturalHeight;
  src.getContext('2d').drawImage(img, 0, 0);
  const det = await detectQuad(eng, src);
  if (!det) return { canvas: src, method: 'none' };
  const warped = warpCanvas(src, det.quad, WARP_MAX);
  const o = await classifyOrientation(eng, warped);
  const mode = (4 - o.angle / 90) % 4;
  const fin = rotateCanvas(warped, mode);
  return { canvas: fin, method: 'ai', angle: o.angle, oriScore: o.score, fill: det.miss };
}

// ============================================================
// 亮度 / 锐化（纯像素，默认参数零成本；只作用于 A4 用图）
// ============================================================
function applyAdjustments(sourceCanvas, opts) {
  const alpha = 0.5 + opts.brightness / 100;
  if (Math.abs(alpha - 1) <= 0.01 && opts.sharpen <= 0) return sourceCanvas;
  const w = sourceCanvas.width, h = sourceCanvas.height;
  const c = document.createElement('canvas'); c.width = w; c.height = h;
  const ctx = c.getContext('2d');
  ctx.drawImage(sourceCanvas, 0, 0);
  const img = ctx.getImageData(0, 0, w, h);
  const d = img.data;
  if (Math.abs(alpha - 1) > 0.01) {
    for (let i = 0; i < d.length; i += 4) {
      d[i] = d[i] * alpha; d[i + 1] = d[i + 1] * alpha; d[i + 2] = d[i + 2] * alpha;
    }
  }
  if (opts.sharpen > 0) {
    const t = opts.sharpen / 100, ctr = 1 + 4 * t;
    const src = new Uint8ClampedArray(d);
    const at = (x, y) => ((y < 0 ? 0 : y > h - 1 ? h - 1 : y) * w + (x < 0 ? 0 : x > w - 1 ? w - 1 : x)) * 4;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      for (let k = 0; k < 3; k++) {
        d[i + k] = ctr * src[i + k]
          - t * (src[at(x, y - 1) + k] + src[at(x, y + 1) + k] + src[at(x - 1, y) + k] + src[at(x + 1, y) + k]);
      }
    }
  }
  ctx.putImageData(img, 0, 0);
  return c;
}

// ============================================================
// A4 渲染
// ============================================================
function roundedRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}
function drawCard(ctx, card, x, y, rotDeg) {
  ctx.save();
  ctx.translate(x + CARD_W / 2, y + CARD_H / 2);
  ctx.rotate(rotDeg * Math.PI / 180);
  const sc = Math.max(CARD_W / card.width, CARD_H / card.height);
  roundedRect(ctx, -CARD_W / 2, -CARD_H / 2, CARD_W, CARD_H, CORNER_R);
  ctx.clip();
  ctx.drawImage(card, -card.width * sc / 2, -card.height * sc / 2, card.width * sc, card.height * sc);
  ctx.restore();
}
function renderA4() {
  const c = document.getElementById('a4-canvas');
  c.width = A4_W; c.height = A4_H;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, A4_W, A4_H);
  const totalH = CARD_H * 2 + GAP;
  const startY = Math.round((A4_H - totalH) / 2);
  const x = Math.round((A4_W - CARD_W) / 2);
  if (state.back.canvas)  drawCard(ctx, state.back.canvas, x, startY, state.back.rot);
  if (state.front.canvas) drawCard(ctx, state.front.canvas, x, startY + CARD_H + GAP, state.front.rot);
  updateBtns();
}

// ============================================================
// 上传
// ============================================================
function setupSlot(side, slotId, fileId) {
  const slot = document.getElementById(slotId);
  const fileInput = document.getElementById(fileId);
  slot.addEventListener('click', () => fileInput.click());
  slot.addEventListener('dragover', e => { e.preventDefault(); slot.classList.add('drag'); });
  slot.addEventListener('dragleave', () => slot.classList.remove('drag'));
  slot.addEventListener('drop', e => {
    e.preventDefault();
    slot.classList.remove('drag');
    if (e.dataTransfer.files[0]) handleFile(side, e.dataTransfer.files[0]);
  });
  fileInput.addEventListener('change', e => {
    if (e.target.files[0]) handleFile(side, e.target.files[0]);
    fileInput.value = '';
  });
}

async function handleFile(side, file) {
  const slot = document.getElementById('slot-' + side);
  slot.innerHTML = '<div class="busy"><div class="spinner"></div><span id="busy-text">正在自动拉正…</span></div>';
  try {
    const dataUrl = await new Promise((resolve, reject) => {
      const r = new FileReader();
      r.onload = () => resolve(r.result);
      r.onerror = reject;
      r.readAsDataURL(file);
    });
    const img = await new Promise((resolve, reject) => {
      const i = new Image();
      i.onload = () => resolve(i);
      i.onerror = reject;
      i.src = dataUrl;
    });
    state[side].img = img;
    const result = await correctIdCard(img);
    state[side].rawCanvas = result.canvas;
    state[side].canvas = applyAdjustments(result.canvas, {
      brightness: state.brightness, sharpen: state.sharpen
    });
    state[side].rot = 0;
    slot.innerHTML = '';
    const im = document.createElement('img');
    im.src = state[side].rawCanvas.toDataURL();
    slot.appendChild(im);
    document.getElementById('rot-' + side).style.display = 'flex';
    document.getElementById('rot-' + side + '-val').textContent = '0°';
    renderA4();
    if (result.method === 'none') {
      alert('未自动识别到身份证，请确认照片中证件完整、边缘清晰，或换一张再试');
    }
  } catch (e) {
    console.error(e);
    slot.innerHTML = '<div class="placeholder">处理失败，请换一张</div>';
    alert('图片处理失败：' + (e && e.message ? e.message : e));
  }
}

// ===== 手动旋转（兜底）=====
document.querySelectorAll('.rot-row button').forEach(btn => {
  btn.addEventListener('click', () => {
    const side = btn.dataset.side;
    const d = parseInt(btn.dataset.d);
    state[side].rot = (state[side].rot + d + 360) % 360;
    document.getElementById('rot-' + side + '-val').textContent = state[side].rot + '°';
    renderA4();
  });
});

// ===== 滑块（只重算 A4 用图，槽里拉正缩略图不变）=====
const bInp = document.getElementById('brightness');
const sInp = document.getElementById('sharpen');
bInp.addEventListener('input', () => {
  state.brightness = parseInt(bInp.value);
  document.getElementById('brightness-val').textContent = bInp.value;
  reprocess();
});
sInp.addEventListener('input', () => {
  state.sharpen = parseInt(sInp.value);
  document.getElementById('sharpen-val').textContent = sInp.value;
  reprocess();
});
let reprocessTimer = null;
function reprocess() {
  clearTimeout(reprocessTimer);
  reprocessTimer = setTimeout(() => {
    const opts = { brightness: state.brightness, sharpen: state.sharpen };
    for (const side of ['front', 'back']) {
      if (state[side].rawCanvas) state[side].canvas = applyAdjustments(state[side].rawCanvas, opts);
    }
    renderA4();
  }, 120);
}

// ===== 按钮 =====
function updateBtns() {
  const has = state.front.canvas || state.back.canvas;
  document.getElementById('btn-print').disabled = !has;
  document.getElementById('btn-download').disabled = !has;
}
document.getElementById('btn-print').addEventListener('click', () => window.print());
document.getElementById('btn-download').addEventListener('click', () => {
  const canvas = document.getElementById('a4-canvas');
  // 桌面 exe（Electron）：走系统“另存为”对话框
  if (window.exeAPI && window.exeAPI.saveImage) {
    window.exeAPI.saveImage(canvas.toDataURL('image/png'), '身份证A4拼版').then(r => {
      if (r && !r.ok && !r.canceled && r.msg) alert('保存失败：' + r.msg);
    });
    return;
  }
  // 普通浏览器：走浏览器下载
  canvas.toBlob(blob => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = '身份证A4拼版.png'; a.click();
    URL.revokeObjectURL(url);
  }, 'image/png');
});

// ===== 重置：清空两面、复位亮度/锐化、恢复初始引导底图（不重载引擎，瞬时完成）=====
function resetAll() {
  for (const side of ['front', 'back']) {
    state[side].img = null;
    state[side].canvas = null;
    state[side].rawCanvas = null;
    state[side].rot = 0;
    const fi = document.getElementById('file-' + side);
    if (fi) fi.value = '';
  }
  state.brightness = 50;
  state.sharpen = 0;
  bInp.value = '50';
  sInp.value = '0';
  document.getElementById('brightness-val').textContent = '50';
  document.getElementById('sharpen-val').textContent = '0';
  loadExample('front', 'assets/images/f.png?v=5');
  loadExample('back', 'assets/images/z.png?v=5');
}
document.getElementById('btn-reset').addEventListener('click', resetAll);

// ===== 初始化（首屏只载轻量 UI；引擎在浏览器空闲时于后台预热，首次上传通常已就绪）=====
setupSlot('front', 'slot-front', 'file-front');
setupSlot('back', 'slot-back', 'file-back');
renderA4();

// 后台预热：页面资源加载完、浏览器空闲后，静默加载并初始化引擎，
// 把首次约几秒的加载成本移到用户挑选照片之前，上传时几乎零等待。
function scheduleWarmup() {
  const start = () => { window.__warmStartAt = Math.round(performance.now()); ensureEngine(true).catch(() => {}); };
  if ('requestIdleCallback' in window) requestIdleCallback(start, { timeout: 2000 });
  else setTimeout(start, 600);
}
window.addEventListener('load', () => setTimeout(scheduleWarmup, 300));

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const i = new Image();
    i.onload = () => resolve(i);
    i.onerror = reject;
    i.src = src;
  });
}
async function loadExample(side, src) {
  try {
    const img = await loadImage(src);
    state[side].img = img;
    state[side].canvas = null;
    state[side].rot = 0;
    const slot = document.getElementById('slot-' + side);
    slot.innerHTML = '';
    const im = document.createElement('img');
    im.src = src;
    slot.appendChild(im);
    document.getElementById('rot-' + side).style.display = 'flex';
    document.getElementById('rot-' + side + '-val').textContent = '0°';
    renderA4();
  } catch (e) { console.error(e); }
}
loadExample('front', 'assets/images/f.png?v=5');
loadExample('back', 'assets/images/z.png?v=5');
