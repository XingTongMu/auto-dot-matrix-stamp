// components/image-cropper/image-cropper.js
const windowInfo = wx.getWindowInfo();
const PR = windowInfo.pixelRatio || 2;
const WW = windowInfo.windowWidth;

Component({
  properties: {
    src: { type: String, value: '' },
    show: { type: Boolean, value: false },
  },

  data: {
    cx: 0, cy: 0, cs: 200, // crop x/y/size
    ix: 0, iy: 0, iw: 0, ih: 0, // image display pos/size
  },

  _c: null, _ctx: null, _img: null,
  _cl: 0, _ct: 0, // canvas left/top
  _drag: null, _pinch: null,

  observers: {
    show(s) {
      console.log('[c] show:', s);
      if (!s) { this._img = null; return; }
      if (this.properties.src) {
        setTimeout(() => this._go(), 200);
      }
    },
  },

  ready() {
    console.log('[c] ready');
    // 预初始化 canvas 节点和位置
    this._getCanvas();
  },

  methods: {
    _getCanvas() {
      this.createSelectorQuery().select('#cropCanvas').fields({ node: true, size: true }).exec(r => {
        console.log('[c] getCanvas:', r ? 'ok' : 'fail');
        if (r && r[0] && r[0].node) {
          const c = r[0].node;
          const ctx = c.getContext('2d');
          const ww = 375;
          c.width = ww * PR; c.height = ww * PR;
          ctx.scale(PR, PR);
          this._c = c;
          this._ctx = ctx;
          // 获取位置
          this.createSelectorQuery().select('#cropCanvas').boundingClientRect(rect => {
            if (rect) { this._cl = rect.left; this._ct = rect.top; }
          }).exec();
        }
      });
    },

    _go() {
      if (!this._c || !this.properties.show || !this.properties.src) {
        setTimeout(() => this._go(), 200);
        return;
      }
      console.log('[c] go');
      this._load(this.properties.src);
    },

    _load(src) {
      console.log('[c] load:', src);
      const img = this._c.createImage();
      img.onload = () => {
        console.log('[c] loaded:', img.width, img.height);
        this._img = img;
        const w = 375;
        let dw, dh;
        if (img.width / img.height > 1) { dw = w; dh = w / img.width * img.height; }
        else { dh = w; dw = w / img.height * img.width; }
        const dx = (w - dw) / 2, dy = (w - dh) / 2;
        let s = Math.min(dw, dh) * 0.8;
        s = Math.max(80, Math.min(s, Math.min(dw, dh)));
        const cx = dx + (dw - s) / 2, cy = dy + (dh - s) / 2;
        this.setData({ ix: dx, iy: dy, iw: dw, ih: dh, cx: Math.round(cx), cy: Math.round(cy), cs: Math.round(s) });
        this._draw();
      };
      img.onerror = () => {};
      img.src = src;
    },

    _draw() {
      const ctx = this._ctx, c = this._c, img = this._img;
      if (!ctx || !c || !img) return;
      const { cx, cy, cs, ix, iy, iw, ih } = this.data;
      const w = 375;
      c.width = w * PR; c.height = w * PR;
      ctx.scale(PR, PR);
      ctx.clearRect(0, 0, w, w);
      ctx.drawImage(img, ix, iy, iw, ih);
      ctx.fillStyle = 'rgba(0,0,0,0.5)';
      ctx.fillRect(0, 0, w, cy);
      ctx.fillRect(0, cy+cs, w, w-cy-cs);
      ctx.fillRect(0, cy, cx, cs);
      ctx.fillRect(cx+cs, cy, w-cx-cs, cs);
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 2;
      ctx.strokeRect(cx, cy, cs, cs);
      ctx.lineWidth = 5; ctx.shadowColor = 'rgba(0,0,0,0.3)'; ctx.shadowBlur = 4;
      const G=2, L=20;
      ctx.beginPath(); ctx.moveTo(cx+G, cy+G); ctx.lineTo(cx+G, cy+G+L); ctx.moveTo(cx+G, cy+G); ctx.lineTo(cx+G+L, cy+G); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(cx+cs-G, cy+G); ctx.lineTo(cx+cs-G, cy+G+L); ctx.moveTo(cx+cs-G, cy+G); ctx.lineTo(cx+cs-G-L, cy+G); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(cx+G, cy+cs-G); ctx.lineTo(cx+G, cy+cs-G-L); ctx.moveTo(cx+G, cy+cs-G); ctx.lineTo(cx+G+L, cy+cs-G); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(cx+cs-G, cy+cs-G); ctx.lineTo(cx+cs-G, cy+cs-G-L); ctx.moveTo(cx+cs-G, cy+cs-G); ctx.lineTo(cx+cs-G-L, cy+cs-G); ctx.stroke();
      ctx.shadowBlur = 0;
    },

    onTouchStart(e) {
      if (!this._img) return;
      if (e.touches.length === 1) {
        this._drag = { x: e.touches[0].clientX - this._cl, y: e.touches[0].clientY - this._ct, cx: this.data.cx, cy: this.data.cy };
      } else if (e.touches.length === 2) {
        const dx = e.touches[1].clientX - e.touches[0].clientX;
        const dy = e.touches[1].clientY - e.touches[0].clientY;
        this._pinch = { dist: Math.sqrt(dx*dx+dy*dy), cs: this.data.cs };
      }
    },

    onTouchMove(e) {
      if (!this._img) return;
      if (this._drag && e.touches.length === 1) {
        const nx = e.touches[0].clientX - this._cl, ny = e.touches[0].clientY - this._ct;
        let x = this._drag.cx + (nx - this._drag.x), y = this._drag.cy + (ny - this._drag.y);
        const { ix, iy, iw, ih, cs } = this.data;
        x = Math.max(ix, Math.min(x, ix+iw-cs));
        y = Math.max(iy, Math.min(y, iy+ih-cs));
        this.setData({ cx: Math.round(x), cy: Math.round(y) });
        this._draw();
      } else if (this._pinch && e.touches.length === 2) {
        const dx = e.touches[1].clientX - e.touches[0].clientX;
        const dy = e.touches[1].clientY - e.touches[0].clientY;
        const d = Math.sqrt(dx*dx+dy*dy);
        if (this._pinch.dist <= 0) return;
        let s = Math.round(this._pinch.cs * d / this._pinch.dist);
        const { ix, iy, iw, ih } = this.data;
        s = Math.max(80, Math.min(s, Math.min(iw, ih)));
        const os = this.data.cs;
        const cx_ = this.data.cx + os/2, cy_ = this.data.cy + os/2;
        let x = Math.round(cx_ - s/2), y = Math.round(cy_ - s/2);
        x = Math.max(ix, Math.min(x, ix+iw-s));
        y = Math.max(iy, Math.min(y, iy+ih-s));
        this.setData({ cs: s, cx: x, cy: y });
        this._draw();
      }
    },

    onTouchEnd() { this._drag = null; this._pinch = null; },

    onCancel() { this.triggerEvent('croppercancel'); },

    onConfirm() {
      const that = this;
      const { _c: c, _ctx: ctx, _img: img } = this;
      if (!c || !ctx || !img) { wx.showToast({ title: '裁剪失败', icon: 'none' }); return; }
      const { cx, cy, cs, ix, iy, iw, ih } = this.data;
      const scale = img.width / iw;
      const ow = Math.round(cs * scale);
      wx.showLoading({ title: '裁剪中...' });
      const w = 375;
      c.width = w * PR; c.height = w * PR;
      ctx.scale(PR, PR);
      ctx.clearRect(0, 0, w, w);
      ctx.drawImage(img, ix, iy, iw, ih);
      wx.canvasToTempFilePath({
        x: cx, y: cy, width: cs, height: cs,
        destWidth: ow, destHeight: ow,
        canvas: c, fileType: 'png', quality: 1,
        success(res) { wx.hideLoading(); that.triggerEvent('cropperconfirm', { tempFilePath: res.tempFilePath, width: ow, height: ow }); },
        fail(e) { wx.hideLoading(); console.log(e); wx.showToast({ title: '裁剪失败', icon: 'none' }); },
      });
    },
  },
});
