/**
 * ink-splash.js —— 泼墨（油漆桶）功能组件
 *
 * 功能：
 *   1. 在线稿 Canvas 上点击空白区域 → 泛洪填充为黑色
 *   2. 闭合检测：填充触碰到图像边缘则拒绝
 *   3. 多步撤回
 *
 * 方案：
 *   使用 catchtouchstart + catchtouchend 检测「点击」
 *   boundingClientRect 实时查询用于坐标换算
 */

const UNDO_MAX = 30;
const FILL_TOLERANCE = 35;
const LINE_THRESHOLD = 100;
const PREVIEW_R = 200; // 预览填充色：浅灰
const PREVIEW_G = 200;
const PREVIEW_B = 200;

Component({
  properties: {
    src: { type: String, value: '' },
    visible: { type: Boolean, value: false },
  },

  data: {
    canUndo: false,
    hintText: '👆 点击线稿空白区域 → 泼墨填充为黑色',
    fillCount: 0,
    show: false,
    // 悬停预览
    showPaintCursor: false,
    cursorPageX: 0,
    cursorPageY: 0,
  },

  lifetimes: {
    attached() {
      this._reset();
    },
    detached() {
      this._clean();
    },
  },

  observers: {
    'src, visible': function (src, v) {
      if (src && v) {
        this.setData({ show: true });
        setTimeout(() => this._initCanvas(), 300);
      } else {
        this.setData({ show: false });
      }
    },
  },

  methods: {

    _reset() {
      this._canvas = null;
      this._ctx = null;
      this._img = null;
      this._ready = false;
      this._stack = [];
      this._current = null;
      this._cssW = 300;
      this._cssH = 300;
      this._canvasRect = null;
      this._previewing = false;
      this._previewPixels = null;
      this._lastPx = -1;
      this._lastPy = -1;
    },

    _clean() {
      this._img = null;
      this._canvas = null;
      this._ctx = null;
      this._stack = [];
      this._current = null;
    },

    /* =================================================================
     * 初始化
     * ================================================================= */
    _initCanvas() {
      const that = this;
      this.createSelectorQuery()
        .select('#inkCanvas')
        .fields({ node: true, size: true })
        .exec(r => {
          if (!r || !r[0] || !r[0].node) {
            setTimeout(() => that._initCanvas(), 500);
            return;
          }

          const canvas = r[0].node;
          const ctx = canvas.getContext('2d');
          that._canvas = canvas;
          that._ctx = ctx;

          // 缓存 Canvas CSS 显示尺寸
          that._cssW = r[0].width || 300;
          that._cssH = r[0].height || 300;

          const img = canvas.createImage();
          img.onload = () => {
            that._img = img;
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0, img.width, img.height);

            try {
              const id = ctx.getImageData(0, 0, img.width, img.height);
              that._current = id;
              that._stack = [id];
            } catch (e) {
              wx.showToast({ title: '泼墨加载失败', icon: 'none' });
              return;
            }

            that._ready = true;
            that.setData({
              canUndo: false,
              fillCount: 0,
              hintText: '👆 点击线稿空白区域 → 泼墨填充为黑色',
            });
          };
          img.onerror = () => wx.showToast({ title: '泼墨: 图片加载失败', icon: 'none' });
          img.src = that.properties.src;
        });
    },

    /* =================================================================
     * 刷新 Canvas 位置缓存（用于 page→pixel 坐标换算）
     * ================================================================= */
    _refreshRect() {
      const that = this;
      this.createSelectorQuery()
        .select('#inkCanvas')
        .boundingClientRect(rect => {
          if (rect) {
            that._canvasRect = rect;
            that._cssW = rect.width;
            that._cssH = rect.height;
          }
        })
        .exec();
    },

    /* =================================================================
     * 触控开始：刷新位置缓存 + 隐藏油漆桶光标
     * ================================================================= */
    onTouchStart(e) {
      if (!this._ready) return;
      this._refreshRect();
      this.setData({ showPaintCursor: false });
    },

    /* =================================================================
     * 触控移动：显示悬停预览 + 油漆桶光标
     * ================================================================= */
    onTouchMove(e) {
      if (!this._ready || !this._current || !this._img) return;
      const t = e.touches && e.touches[0];
      if (!t) return;

      // 显示油漆桶光标在手指上方
      this.setData({
        showPaintCursor: true,
        cursorPageX: t.pageX,
        cursorPageY: t.pageY - 80,
      });

      // 检测并显示预览
      const rect = this._canvasRect;
      if (!rect || !rect.width || !rect.height) return;

      const relX = t.pageX - rect.left;
      const relY = t.pageY - rect.top;
      if (relX < 0 || relY < 0 || relX > rect.width || relY > rect.height) {
        this._clearPreview();
        return;
      }

      const imgW = this._img.width;
      const imgH = this._img.height;
      let px = Math.floor(relX * imgW / rect.width);
      let py = Math.floor(relY * imgH / rect.height);
      px = Math.max(0, Math.min(imgW - 1, px));
      py = Math.max(0, Math.min(imgH - 1, py));

      // 位置未变则跳过
      if (px === this._lastPx && py === this._lastPy) return;
      this._lastPx = px;
      this._lastPy = py;

      this._showPreviewAt(px, py);
    },

    /* =================================================================
     * 触控结束：清除预览 + 隐藏光标
     * ================================================================= */
    onTouchEnd() {
      this._clearPreview();
      this._lastPx = -1;
      this._lastPy = -1;
      this.setData({ showPaintCursor: false });
    },

    /* =================================================================
     * 在指定像素位置预览（浅灰色填充）
     * ================================================================= */
    _showPreviewAt(px, py) {
      const W = this._img.width;
      const H = this._img.height;
      const data = this._current.data;

      const si = (py * W + px) * 4;
      const sr = data[si], sg = data[si + 1], sb = data[si + 2];
      const gray = 0.299 * sr + 0.587 * sg + 0.114 * sb;

      if (gray < LINE_THRESHOLD) {
        this._clearPreview();
        return;
      }

      const src = new Uint8ClampedArray(data);
      const vis = new Uint8Array(W * H);
      const q = [[px, py]];
      let h = 0, hit = false;
      const pixels = [];

      while (h < q.length) {
        const [x, y] = q[h++];
        if (x < 0 || x >= W || y < 0 || y >= H) { hit = true; continue; }
        const vi = y * W + x;
        if (vis[vi]) continue;
        vis[vi] = 1;
        const ii = vi * 4;
        if (Math.abs(src[ii] - sr) > FILL_TOLERANCE ||
            Math.abs(src[ii + 1] - sg) > FILL_TOLERANCE ||
            Math.abs(src[ii + 2] - sb) > FILL_TOLERANCE) continue;
        pixels.push(vi);
        q.push([x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1]);
      }

      if (hit || pixels.length === 0) {
        this._clearPreview();
        return;
      }

      // 可填充 — 显示浅灰色预览
      const previewData = new Uint8ClampedArray(data);
      for (const vi of pixels) {
        const ii = vi * 4;
        previewData[ii] = PREVIEW_R;
        previewData[ii + 1] = PREVIEW_G;
        previewData[ii + 2] = PREVIEW_B;
        previewData[ii + 3] = 255;
      }

      this._ctx.putImageData(new ImageData(previewData, W, H), 0, 0);
      this._previewing = true;
      this._previewPixels = pixels;
    },

    /* =================================================================
     * 清除预览（恢复 current 状态）
     * ================================================================= */
    _clearPreview() {
      if (!this._previewing || !this._ctx || !this._current) return;
      this._ctx.putImageData(this._current, 0, 0);
      this._previewing = false;
      this._previewPixels = null;
    },

    /* =================================================================
     * Canvas 点击 → 计算内部像素坐标 → 填充
     * ================================================================= */
    onCanvasTap(e) {
      if (!this._ready || !this._current || !this._img) return;

      // 清除预览
      this._clearPreview();

      // type="2d" canvas tap: e.detail = {x, y} (canvas-relative CSS px)
      let relX, relY;

      if (e.detail && typeof e.detail.x === 'number') {
        relX = e.detail.x;
        relY = e.detail.y;
      }

      if (relX === undefined) {
        // 兜底：从 changedTouches 计算
        const ct = e.changedTouches && e.changedTouches[0];
        if (!ct) return;

        const that = this;
        this.createSelectorQuery()
          .select('#inkCanvas')
          .boundingClientRect(rect => {
            if (!rect || !rect.width || !rect.height) return;
            that._cssW = rect.width;
            that._cssH = rect.height;
            const rx = ct.pageX - rect.left;
            const ry = ct.pageY - rect.top;
            that._doFill(rx, ry);
          })
          .exec();
        return;
      }

      this._doFill(relX, relY);
    },

    /* =================================================================
     * 将 CSS 相对坐标映射到内部像素坐标并执行填充
     * ================================================================= */
    _doFill(relX, relY) {
      const imgW = this._img.width;
      const imgH = this._img.height;

      let ix = Math.floor(relX * imgW / this._cssW);
      let iy = Math.floor(relY * imgH / this._cssH);
      ix = Math.max(0, Math.min(imgW - 1, ix));
      iy = Math.max(0, Math.min(imgH - 1, iy));

      this._floodFill(ix, iy);
    },

    /* =================================================================
     * 泛洪填充
     * ================================================================= */
    _floodFill(sx, sy) {
      const W = this._img.width;
      const H = this._img.height;
      const src = new Uint8ClampedArray(this._current.data);
      const dst = this._current.data;

      const si = (sy * W + sx) * 4;
      const sr = src[si], sg = src[si + 1], sb = src[si + 2];
      const gray = 0.299 * sr + 0.587 * sg + 0.114 * sb;

      if (gray < LINE_THRESHOLD) {
        wx.showToast({ title: '⚠️ 请点击空白区域', icon: 'none' });
        return;
      }

      const vis = new Uint8Array(W * H);
      const q = [[sx, sy]];
      let h = 0, hit = false;
      const pixels = [];

      while (h < q.length) {
        const [x, y] = q[h++];
        if (x < 0 || x >= W || y < 0 || y >= H) { hit = true; continue; }
        const vi = y * W + x;
        if (vis[vi]) continue;
        vis[vi] = 1;
        const ii = vi * 4;
        if (Math.abs(src[ii] - sr) > FILL_TOLERANCE ||
            Math.abs(src[ii + 1] - sg) > FILL_TOLERANCE ||
            Math.abs(src[ii + 2] - sb) > FILL_TOLERANCE) continue;
        pixels.push(vi);
        q.push([x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1]);
      }

      if (hit) { wx.showToast({ title: '⚠️ 区域未闭合，无法填充', icon: 'none' }); return; }
      if (!pixels.length) { wx.showToast({ title: '⚠️ 无可填充像素', icon: 'none' }); return; }

      // 保存快照
      this._stack.push(new ImageData(new Uint8ClampedArray(dst), W, H));
      if (this._stack.length > UNDO_MAX) this._stack.shift();

      // 涂黑
      for (const vi of pixels) {
        const ii = vi * 4;
        dst[ii] = 0; dst[ii + 1] = 0; dst[ii + 2] = 0; dst[ii + 3] = 255;
      }

      this._ctx.putImageData(this._current, 0, 0);
      this.setData({
        canUndo: this._stack.length > 1,
        fillCount: this.data.fillCount + 1,
        hintText: `✅ 已填充 ${pixels.length} 像素 | 「↩」可撤回`,
      });
    },

    /* =================================================================
     * 撤回
     * ================================================================= */
    onUndo() {
      if (this._stack.length <= 1) {
        wx.showToast({ title: '没有可撤回的操作', icon: 'none' });
        return;
      }
      this._stack.pop();
      const prev = this._stack[this._stack.length - 1];
      this._current = new ImageData(new Uint8ClampedArray(prev.data), prev.width, prev.height);
      this._ctx.putImageData(this._current, 0, 0);
      this.setData({
        canUndo: this._stack.length > 1,
        fillCount: this.data.fillCount - 1,
        hintText: '↩ 已撤回 | 可继续点击填充',
      });
    },
  },
});
