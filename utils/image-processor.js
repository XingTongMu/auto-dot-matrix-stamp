/* ============================================================
   AI智能印章 - 图像处理模块 (utils/image-processor.js)

   三个独立管线，纯数据进/出，零外部依赖：
     processToDither()   — 照片复古模式（Floyd-Steinberg 抖动）
     processToEdge()     — 硬朗线稿模式（Sobel + 膨胀 + 二值化）
     processToHalftone() — 波普图腾模式（Cluster-dot 有序抖动）

   输入：Uint8ClampedArray imageData, number w, number h
   输出：Uint8Array(1024) → 0=凹/白, 1=凸/黑
   ============================================================ */

const GRID_SIZE = 32;
const TOTAL_PIXELS = GRID_SIZE * GRID_SIZE;

// ============================================================
// 共享工具
// ============================================================

/** ITU-R BT.601 灰度化 → Float32Array(w*h)，范围 0-255 */
function toGrayscale(data, w, h) {
  const gray = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const idx = i * 4;
    gray[i] = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
  }
  return gray;
}

/** 区域平均降采样 → Float32Array(outSize*outSize)，范围 0-255 */
function downscale(src, srcW, srcH, outSize) {
  const result = new Float32Array(outSize * outSize);
  const bw = srcW / outSize;
  const bh = srcH / outSize;
  for (let row = 0; row < outSize; row++) {
    for (let col = 0; col < outSize; col++) {
      const sx0 = Math.floor(col * bw);
      const sy0 = Math.floor(row * bh);
      const sx1 = Math.min(Math.floor((col + 1) * bw), srcW);
      const sy1 = Math.min(Math.floor((row + 1) * bh), srcH);
      let sum = 0, cnt = 0;
      for (let y = sy0; y < sy1; y++) {
        for (let x = sx0; x < sx1; x++) {
          sum += src[y * srcW + x];
          cnt++;
        }
      }
      result[row * outSize + col] = cnt > 0 ? sum / cnt : 128;
    }
  }
  return result;
}

/** 统计黑色像素数 */
export function countBlack(pixelArray) {
  let n = 0;
  for (let i = 0; i < pixelArray.length; i++) {
    if (pixelArray[i] === 1) n++;
  }
  return n;
}

// ============================================================
// 管线 1：照片复古模式 — Floyd-Steinberg 误差扩散抖动
// ============================================================

/**
 * processToDither(imageData, w, h)
 *
 * 灰度化 → 缩放至 32×32 → Floyd-Steinberg 误差扩散
 *
 * 物理映射（无歧义）：
 *   暗像素 → stamp 1（凸起/有墨）   亮像素 → stamp 0（凹下/无墨）
 *   量化阈值 = 128（灰度中点），误差按 7/16 3/16 5/16 1/16 向右下扩散
 *
 * @param {Uint8ClampedArray} imageData  RGBA 原始像素
 * @param {number} w  图像宽度
 * @param {number} h  图像高度
 * @returns {Uint8Array} 1024 个 0/1
 */
export function processToDither(imageData, w, h) {
  const gray = toGrayscale(imageData, w, h);
  const scaled = downscale(gray, w, h, GRID_SIZE);

  // Floyd-Steinberg 误差扩散
  // 核心原理：逐像素量化（≥128→白/255，<128→黑/0），
  // 将量化误差按固定比例扩散到尚未处理的右/下/左下/右下邻居，
  // 使得局部平均灰度逼近原图 —— 用黑白点阵的疏密模拟连续调。
  const pixels = new Float32Array(scaled);
  const out = new Uint8Array(TOTAL_PIXELS);

  for (let row = 0; row < GRID_SIZE; row++) {
    for (let col = 0; col < GRID_SIZE; col++) {
      const i = row * GRID_SIZE + col;
      const old = pixels[i];

      // 量化判定
      const q = old >= 128 ? 255 : 0;      // 量化值（色彩空间）
      out[i] = q === 0 ? 1 : 0;            // 输出值（stamp：0=凹/白, 1=凸/黑）

      // 误差扩散至四个未处理邻居
      const err = old - q;
      if (col + 1 < GRID_SIZE) pixels[i + 1]                += err * 7 / 16;
      if (row + 1 < GRID_SIZE && col > 0) pixels[i + GRID_SIZE - 1] += err * 3 / 16;
      if (row + 1 < GRID_SIZE)              pixels[i + GRID_SIZE]     += err * 5 / 16;
      if (row + 1 < GRID_SIZE && col + 1 < GRID_SIZE) pixels[i + GRID_SIZE + 1] += err * 1 / 16;
    }
  }

  return out;
}

// ============================================================
// 管线 2：硬朗线稿模式 — Sobel + 形态学膨胀 + 二值化
// ============================================================

/** 3×3 高斯模糊，kernel [[1,2,1],[2,4,2],[1,2,1]] / 16 */
function gaussianBlur(data, w, h) {
  const ker = [[1, 2, 1], [2, 4, 2], [1, 2, 1]];
  const out = new Uint8ClampedArray(data.length);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let r = 0, g = 0, b = 0;
      for (let ky = -1; ky <= 1; ky++) {
        for (let kx = -1; kx <= 1; kx++) {
          const px = Math.min(Math.max(x + kx, 0), w - 1);
          const py = Math.min(Math.max(y + ky, 0), h - 1);
          const idx = (py * w + px) * 4;
          const k = ker[ky + 1][kx + 1];
          r += data[idx] * k;
          g += data[idx + 1] * k;
          b += data[idx + 2] * k;
        }
      }
      const oi = (y * w + x) * 4;
      out[oi]     = Math.round(r / 16);
      out[oi + 1] = Math.round(g / 16);
      out[oi + 2] = Math.round(b / 16);
      out[oi + 3] = data[oi + 3];
    }
  }
  return out;
}

/** Sobel 边缘检测 → 二值边缘图 (1=边缘, 0=非边缘)，自适应阈值 = maxMagnitude * 0.12 */
function sobelBinaryEdge(data, w, h) {
  // alpha 加权灰度化
  const gray = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const idx = i * 4;
    const a = data[idx + 3] / 255;
    gray[i] = (0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2]) * a;
  }

  // Sobel 梯度幅值
  const mag = new Float32Array(w * h);
  let maxMag = 0;
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
      const m = Math.sqrt(gx * gx + gy * gy);
      mag[y * w + x] = m;
      if (m > maxMag) maxMag = m;
    }
  }

  const thresh = maxMag * 0.12;
  const bin = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) {
    bin[i] = mag[i] > thresh ? 1 : 0;
  }
  return bin;
}

/** 形态学膨胀 — 3×3 方形结构元素，iterations 次迭代 */
function dilate(binary, w, h, iterations) {
  let src = binary;
  for (let iter = 0; iter < iterations; iter++) {
    const dst = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        if (src[y * w + x] === 1) {
          dst[y * w + x] = 1;
          continue;
        }
        // 3×3 邻域内有前景像素 → 膨胀为前景
        let fg = 0;
        for (let dy = -1; dy <= 1 && !fg; dy++) {
          for (let dx = -1; dx <= 1 && !fg; dx++) {
            const ny = y + dy, nx = x + dx;
            if (ny >= 0 && ny < h && nx >= 0 && nx < w && src[ny * w + nx] === 1) {
              fg = 1;
            }
          }
        }
        dst[y * w + x] = fg;
      }
    }
    src = dst;
  }
  return src;
}

/**
 * processToEdge(imageData, w, h)
 *
 * 高斯模糊 → Sobel 边缘检测 → 形态学膨胀加粗 → 缩放至 32×32 → 二值化
 *
 * 关键设计：在原始分辨率检测边缘并膨胀加粗，确保线条在剧烈降采样
 * （如 4000px→32px，125× 压缩）后不会断裂消失。膨胀迭代次数根据
 * 原图尺寸自适应计算。
 *
 * @param {Uint8ClampedArray} imageData  RGBA 原始像素
 * @param {number} w  图像宽度
 * @param {number} h  图像高度
 * @returns {Uint8Array} 1024 个 0/1
 */
export function processToEdge(imageData, w, h) {
  // Step 1: 高斯模糊去噪
  const blurred = gaussianBlur(imageData, w, h);

  // Step 2: Sobel → 二值边缘图
  const edgeBin = sobelBinaryEdge(blurred, w, h);

  // Step 3: 形态学膨胀加粗
  //   膨胀次数 = max(1, min(⌊longEdge / 128⌋, 12))
  //   原图 512px → 4 次；原图 1024px → 8 次；原图 2000px+ → 12 次
  const longEdge = Math.max(w, h);
  const dilateIter = Math.max(1, Math.min(Math.floor(longEdge / 128), 12));
  const dilated = dilate(edgeBin, w, h, dilateIter);

  // Step 4: 缩放至 32×32（用浮点灰度表示边缘密度）
  const edgeGray = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) edgeGray[i] = dilated[i] * 255;
  const scaled = downscale(edgeGray, w, h, GRID_SIZE);

  // Step 5: 二值化输出（阈值 64 — 格内有边缘即输出黑）
  const out = new Uint8Array(TOTAL_PIXELS);
  for (let i = 0; i < TOTAL_PIXELS; i++) {
    out[i] = scaled[i] > 64 ? 1 : 0;
  }
  return out;
}

// ============================================================
// 管线 3：波普图腾模式 — Cluster-dot 有序抖动
// ============================================================

/**
 * 生成 Cluster-dot 聚点阈值矩阵
 *
 * 从 tile 中心向外螺旋排列阈值：
 *   中心阈值最低 → 暗区先出现小黑点 →
 *   灰度加深时黑点向外扩大成圆斑 →
 *   人眼将聚点圆斑整合为连续灰度，产生半色调印刷的波普效果。
 */
function buildClusterMatrix(size) {
  const total = size * size;
  const positions = [];
  const cx = (size - 1) / 2;
  const cy = (size - 1) / 2;

  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      positions.push({ r, c, dist: Math.hypot(r - cx, c - cy) });
    }
  }

  positions.sort((a, b) => {
    const d = a.dist - b.dist;
    if (Math.abs(d) < 1e-9) {
      return Math.atan2(a.r - cx, a.c - cy) - Math.atan2(b.r - cx, b.c - cy);
    }
    return d;
  });

  const m = new Array(size);
  for (let r = 0; r < size; r++) {
    m[r] = new Float32Array(size);
  }
  for (let k = 0; k < positions.length; k++) {
    const { r, c } = positions[k];
    m[r][c] = Math.round((k / (total - 1)) * 255);
  }
  return m;
}

// 8×8 聚点矩阵 — 在 32×32 网格中平铺 4×4 = 16 个 tile
const CLUSTER_SIZE = 8;
const CLUSTER_MATRIX = buildClusterMatrix(CLUSTER_SIZE);

/**
 * processToHalftone(imageData, w, h)
 *
 * 灰度化 → 缩放至 32×32 → Cluster-dot 有序抖动
 *
 * 与 Floyd-Steinberg 的随机噪点不同，有序抖动产生规则的几何纹理。
 * 聚点式（cluster-dot）在暗区形成圆形墨点，亮区留白，类似报纸印刷
 * 或波普艺术的半色调效果。
 *
 * @param {Uint8ClampedArray} imageData  RGBA 原始像素
 * @param {number} w  图像宽度
 * @param {number} h  图像高度
 * @returns {Uint8Array} 1024 个 0/1
 */
export function processToHalftone(imageData, w, h) {
  const gray = toGrayscale(imageData, w, h);
  const scaled = downscale(gray, w, h, GRID_SIZE);

  const out = new Uint8Array(TOTAL_PIXELS);
  for (let row = 0; row < GRID_SIZE; row++) {
    for (let col = 0; col < GRID_SIZE; col++) {
      const i = row * GRID_SIZE + col;
      const threshold = CLUSTER_MATRIX[row % CLUSTER_SIZE][col % CLUSTER_SIZE];

      // 像素灰度 < 阈值 → 黑(1)，灰度 >= 阈值 → 白(0)
      out[i] = scaled[i] < threshold ? 1 : 0;
    }
  }
  return out;
}

// ============================================================
// 公共导出
// ============================================================

export { GRID_SIZE, TOTAL_PIXELS };
