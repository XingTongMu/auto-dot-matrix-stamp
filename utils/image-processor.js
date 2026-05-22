/* ============================================================
   AI智能印章 - 图像处理模块 (utils/image-processor.js)
   支持三模式：dither / edge-aware / edge

   算法参考：
   - Floyd-Steinberg 误差扩散抖动
   - CIEDE2000 色差公式 (Bead-Pattern-Maker)
   - 边缘感知降采样 (Bead-Pattern-Maker edgeAwareDownsample)
   - 二值邻域去噪 (Bead-Pattern-Maker denoisePattern)
   ============================================================ */

const GRID_SIZE = 32;
const TOTAL_PIXELS = GRID_SIZE * GRID_SIZE;

// 边缘感知降采样阈值：区域边缘强度 > 此值则取边缘像素
const EDGE_THRESHOLD = 80;

// ============================================================
// 内部工具函数
// ============================================================

function _loadImage(src, canvas) {
  return new Promise((resolve, reject) => {
    if (!canvas) { reject(new Error('Canvas 不存在')); return; }
    const img = canvas.createImage();
    img.onload = () => resolve(img);
    img.onerror = (e) => reject(e);
    img.src = src;
  });
}

function _gaussianBlur(imageData, w, h) {
  const data = imageData.data;
  const out = new Uint8ClampedArray(data.length);
  const kernel = [[1, 2, 1], [2, 4, 2], [1, 2, 1]];
  const ks = 16;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let r = 0, g = 0, b = 0;
      for (let ky = -1; ky <= 1; ky++) {
        for (let kx = -1; kx <= 1; kx++) {
          const px = Math.min(Math.max(x + kx, 0), w - 1);
          const py = Math.min(Math.max(y + ky, 0), h - 1);
          const idx = (py * w + px) * 4;
          const wt = kernel[ky + 1][kx + 1];
          r += data[idx] * wt;
          g += data[idx + 1] * wt;
          b += data[idx + 2] * wt;
        }
      }
      const oi = (y * w + x) * 4;
      out[oi]     = Math.round(r / ks);
      out[oi + 1] = Math.round(g / ks);
      out[oi + 2] = Math.round(b / ks);
      out[oi + 3] = data[oi + 3];
    }
  }
  return { data: out, width: w, height: h };
}

function _toGrayscale(imageData, w, h) {
  const data = imageData.data;
  const gray = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const idx = i * 4;
    gray[i] = Math.round(0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2]);
  }
  return gray;
}

// ============================================================
// Sobel 边缘检测（直接处理 RGBA，含 alpha 加权）
// 参考 Bead-Pattern-Maker 的实现
// ============================================================
function _sobelEdgeDetectRGBA(imageData, w, h) {
  const data = imageData.data;
  // 灰度化 + alpha 加权
  const gray = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const idx = i * 4;
    const alpha = data[idx + 3] / 255;
    gray[i] = (0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2]) * alpha;
  }

  const edgeMap = new Float32Array(w * h);
  let maxMagnitude = 0;

  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const tl = gray[(y - 1) * w + (x - 1)];
      const tc = gray[(y - 1) * w + x];
      const tr = gray[(y - 1) * w + (x + 1)];
      const ml = gray[y * w + (x - 1)];
      const mr = gray[y * w + (x + 1)];
      const bl = gray[(y + 1) * w + (x - 1)];
      const bc = gray[(y + 1) * w + x];
      const br = gray[(y + 1) * w + (x + 1)];

      const gx = -tl + tr - 2 * ml + 2 * mr - bl + br;
      const gy = -tl - 2 * tc - tr + bl + 2 * bc + br;
      const magnitude = Math.sqrt(gx * gx + gy * gy);
      edgeMap[y * w + x] = magnitude;
      if (magnitude > maxMagnitude) maxMagnitude = magnitude;
    }
  }

  // 归一化到 0~255
  if (maxMagnitude > 0) {
    for (let i = 0; i < edgeMap.length; i++) {
      edgeMap[i] = (edgeMap[i] / maxMagnitude) * 255;
    }
  }
  return edgeMap;
}

function _sobelEdgeDetection(gray, w, h) {
  const edges = new Float32Array(w * h);
  const sobelX = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]];
  const sobelY = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]];
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      let gx = 0, gy = 0;
      for (let ky = -1; ky <= 1; ky++) {
        for (let kx = -1; kx <= 1; kx++) {
          const pv = gray[(y + ky) * w + (x + kx)];
          gx += pv * sobelX[ky + 1][kx + 1];
          gy += pv * sobelY[ky + 1][kx + 1];
        }
      }
      edges[y * w + x] = Math.sqrt(gx * gx + gy * gy);
    }
  }
  return edges;
}

// ============================================================
// 边缘感知降采样（核心新算法）
//
// 对每个目标像素对应的源区域：
//  - 追踪区域内边缘强度最强的像素
//  - 如果最强边缘 > 阈值 → 使用该边缘像素的灰度值（最近邻，保留锐利边缘）
//  - 否则 → 使用区域平均灰度值（保留平滑过渡）
// ============================================================
function _edgeAwareDownsample(gray, edgeMap, srcW, srcH, gridSize) {
  const total = gridSize * gridSize;
  const result = new Uint8Array(total);
  const blockW = srcW / gridSize;
  const blockH = srcH / gridSize;

  for (let row = 0; row < gridSize; row++) {
    for (let col = 0; col < gridSize; col++) {
      const sx0 = Math.floor(col * blockW);
      const sy0 = Math.floor(row * blockH);
      const sx1 = Math.min(Math.floor((col + 1) * blockW), srcW);
      const sy1 = Math.min(Math.floor((row + 1) * blockH), srcH);

      let sum = 0, cnt = 0;
      let maxEdge = 0;
      let edgeGray = 0;

      for (let y = sy0; y < sy1; y++) {
        for (let x = sx0; x < sx1; x++) {
          const gv = gray[y * srcW + x];
          sum += gv;
          cnt++;
          const ev = edgeMap[y * srcW + x];
          if (ev > maxEdge) {
            maxEdge = ev;
            edgeGray = gv;
          }
        }
      }

      const idx = row * gridSize + col;
      if (maxEdge > EDGE_THRESHOLD) {
        // 边缘区域：使用边缘像素的灰度值，保留锐利边界
        result[idx] = edgeGray;
      } else {
        // 平滑区域：使用平均灰度值，保留色调过渡
        result[idx] = Math.round(sum / cnt);
      }
    }
  }
  return result;
}

function _downsampleToGrid(gray, srcW, srcH, gridSize) {
  const result = new Uint8Array(gridSize * gridSize);
  for (let row = 0; row < gridSize; row++) {
    for (let col = 0; col < gridSize; col++) {
      const sx = Math.floor(col * srcW / gridSize);
      const sy = Math.floor(row * srcH / gridSize);
      const ex = Math.floor((col + 1) * srcW / gridSize);
      const ey = Math.floor((row + 1) * srcH / gridSize);
      let sum = 0, cnt = 0;
      for (let y = sy; y < ey; y++) {
        for (let x = sx; x < ex; x++) {
          sum += gray[y * srcW + x];
          cnt++;
        }
      }
      result[row * gridSize + col] = Math.round(sum / cnt);
    }
  }
  return result;
}

function _contrastStretch(arr, len) {
  let minV = 255, maxV = 0;
  for (let i = 0; i < len; i++) {
    if (arr[i] < minV) minV = arr[i];
    if (arr[i] > maxV) maxV = arr[i];
  }
  const range = maxV - minV || 1;
  const result = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    result[i] = Math.round((arr[i] - minV) / range * 255);
  }
  return result;
}

function _floydSteinbergDither(grayValues, gridSize) {
  const total = gridSize * gridSize;
  const THRESHOLD = 128;
  const QUANTUM_WHITE = 255;   // 亮 → 凹下(无墨) → 0
  const QUANTUM_BLACK = 0;     // 暗 → 凸起(有墨) → 1
  const pixels = new Float32Array(grayValues);
  const result = new Uint8Array(total); // 0=白(凹), 1=黑(凸)
  for (let row = 0; row < gridSize; row++) {
    for (let col = 0; col < gridSize; col++) {
      const idx = row * gridSize + col;
      const old = pixels[idx];
      const nv = old >= THRESHOLD ? QUANTUM_WHITE : QUANTUM_BLACK;
      result[idx] = nv === QUANTUM_BLACK ? 1 : 0;
      const err = old - nv;
      if (col + 1 < gridSize) pixels[idx + 1] += err * 7 / 16;
      if (row + 1 < gridSize && col - 1 >= 0) pixels[idx + gridSize - 1] += err * 3 / 16;
      if (row + 1 < gridSize) pixels[idx + gridSize] += err * 5 / 16;
      if (row + 1 < gridSize && col + 1 < gridSize) pixels[idx + gridSize + 1] += err * 1 / 16;
    }
  }
  return result;
}

// ============================================================
// 二值邻域去噪（新功能）
//
// 参考 Bead-Pattern-Maker 的 denoisePattern 思路：
// 对每个像素检查 3×3 邻域，如果它是"孤立异色点"
// （周围同色邻居 < 2 个），则翻转为邻域众数颜色。
// 这能有效消除 Floyd-Steinberg 抖动产生的椒盐噪声。
// ============================================================
function _binaryDenoise(pixelArray, gridSize) {
  const total = gridSize * gridSize;
  const result = new Uint8Array(pixelArray); // 复制

  for (let row = 0; row < gridSize; row++) {
    for (let col = 0; col < gridSize; col++) {
      const idx = row * gridSize + col;
      const me = pixelArray[idx];

      // 收集 3×3 邻域
      let sameCount = 0, totalCount = 0;
      for (let dr = -1; dr <= 1; dr++) {
        for (let dc = -1; dc <= 1; dc++) {
          if (dr === 0 && dc === 0) continue; // 排除自身
          const nr = row + dr;
          const nc = col + dc;
          if (nr < 0 || nr >= gridSize || nc < 0 || nc >= gridSize) continue;
          totalCount++;
          if (pixelArray[nr * gridSize + nc] === me) sameCount++;
        }
      }

      // 如果周围同色邻居不到 2 个，说明是孤立噪点，翻转
      if (totalCount >= 3 && sameCount < 2) {
        result[idx] = me === 1 ? 0 : 1;
      }
    }
  }
  return result;
}

function _countBlack(pixelArray, len) {
  let n = 0;
  for (let i = 0; i < len; i++) {
    if (pixelArray[i] === 1) n++;
  }
  return n;
}

// ============================================================
// 公共辅助：画布上加载图片并返回 ImageData
// ============================================================
async function _loadImageData(canvas, tempFilePath) {
  const img = await _loadImage(tempFilePath, canvas);
  canvas.width = img.width; canvas.height = img.height;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0, img.width, img.height);
  return {
    imgData: ctx.getImageData(0, 0, img.width, img.height),
    W: img.width,
    H: img.height,
  };
}

// ============================================================
// 管线 1：Dither（照片模式）—— 纯灰度 + Floyd-Steinberg 抖动
// ============================================================
async function _ditherPipeline(canvas, tempFilePath, gridSize) {
  const total = gridSize * gridSize;
  const { imgData, W, H } = await _loadImageData(canvas, tempFilePath);

  const blurred = _gaussianBlur(imgData, W, H);
  const gray = _toGrayscale(blurred, W, H);
  const gridGray = _downsampleToGrid(gray, W, H, gridSize);
  const stretched = _contrastStretch(gridGray, total);
  const pixelArray = _floydSteinbergDither(stretched, gridSize);
  const blackCount = _countBlack(pixelArray, total);

  return { pixelArray, blackCount, W, H };
}

// ============================================================
// 管线 2：Edge-Aware（边缘感知模式）—— 推荐用于 Logo/文字/线条
//
// 在降采样时用边缘强度图引导：
//  - 边缘区域 → 最近邻取色 → 保留锐利轮廓
//  - 平滑区域 → 区域平均 → 保留色调过渡
// 最后做二值去噪消除椒盐散点
// ============================================================
async function _edgeAwarePipeline(canvas, tempFilePath, gridSize) {
  const total = gridSize * gridSize;
  const { imgData, W, H } = await _loadImageData(canvas, tempFilePath);

  const blurred = _gaussianBlur(imgData, W, H);
  const gray = _toGrayscale(blurred, W, H);
  const edgeMap = _sobelEdgeDetectRGBA(blurred, W, H);
  const gridGray = _edgeAwareDownsample(gray, edgeMap, W, H, gridSize);
  const stretched = _contrastStretch(gridGray, total);
  let pixelArray = _floydSteinbergDither(stretched, gridSize);

  // 去噪：消除孤立噪点
  pixelArray = _binaryDenoise(pixelArray, gridSize);

  const blackCount = _countBlack(pixelArray, total);
  return { pixelArray, blackCount, W, H };
}

// ============================================================
// 管线 3：Edge（徽章模式）—— Sobel 边缘检测 + 素描融合（保留兼容）
// ============================================================
async function _edgePipeline(canvas, tempFilePath, gridSize) {
  const total = gridSize * gridSize;
  const { imgData, W, H } = await _loadImageData(canvas, tempFilePath);

  const blurred = _gaussianBlur(imgData, W, H);
  const gray = _toGrayscale(blurred, W, H);
  const edges = _sobelEdgeDetection(gray, W, H);

  let maxEdge = 0;
  for (let i = 0; i < W * H; i++) {
    if (edges[i] > maxEdge) maxEdge = edges[i];
  }
  const norm = maxEdge > 0 ? 1 / maxEdge : 1;
  const fused = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) {
    fused[i] = gray[i] * (1 - edges[i] * norm * 0.8);
  }

  const stretched = _contrastStretch(fused, W * H);
  const gridGray = _downsampleToGrid(stretched, W, H, gridSize);
  const finalGray = _contrastStretch(gridGray, total);
  let pixelArray = _floydSteinbergDither(finalGray, gridSize);
  pixelArray = _binaryDenoise(pixelArray, gridSize);

  const blackCount = _countBlack(pixelArray, total);
  return { pixelArray, blackCount, W, H };
}

// ============================================================
// 公开 API
// ============================================================

/**
 * 处理图片并生成二值点阵
 * @param {Object} options
 * @param {string}  options.tempFilePath - 图片临时路径
 * @param {Object}  options.canvas       - 离屏 Canvas 实例
 * @param {string}  [options.mode='dither']   - 'dither' 照片 | 'edge-aware' 边缘感知 | 'edge' 徽章
 * @param {number}  [options.gridSize=32]     - 点阵尺寸
 * @returns {Promise<{pixelArray: Uint8Array, blackCount: number}>}
 */
export async function processImage({
  tempFilePath,
  canvas,
  mode = 'dither',
  gridSize = 32,
}) {
  if (!tempFilePath) throw new Error('缺少图片路径');
  if (!canvas) throw new Error('缺少 Canvas 实例');

  switch (mode) {
    case 'edge-aware':
      return _edgeAwarePipeline(canvas, tempFilePath, gridSize);
    case 'edge':
      return _edgePipeline(canvas, tempFilePath, gridSize);
    default:
      return _ditherPipeline(canvas, tempFilePath, gridSize);
  }
}

export { GRID_SIZE, TOTAL_PIXELS };
