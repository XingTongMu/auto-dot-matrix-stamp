/* ============================================================
   AI智能印章 - 首页逻辑 (pages/index/index.js)
   ⚡ 包含：图像处理 → 点阵生成 → 数据打包 → 蓝牙通信
   ============================================================ */

// ============================================================
// 全局配置常量（只需修改此处即可调整点阵尺寸）
// ============================================================
const GRID_SIZE = 32;                      // 点阵尺寸：32×32（共 1024 像素点）
const TOTAL_PIXELS = GRID_SIZE * GRID_SIZE; // 总像素数 = 1024

// 蓝牙相关常量
const BLE_SERVICE_UUID = '0000FFE0-0000-1000-8000-00805F9B34FB';
const BLE_CHAR_UUID_WRITE = '0000FFE1-0000-1000-8000-00805F9B34FB';
const BLE_CHAR_UUID_NOTIFY = '0000FFE2-0000-1000-8000-00805F9B34FB';
const BLE_DEVICE_NAME_PREFIX = 'TIANJI-STAMP-01';
const MTU_TARGET = 247;

// 通信帧协议常量
const SOF = 0xAA;
const CMD = 0x01;
const EOF = 0xBB;

// 图像处理模块
import { processImage } from '../../utils/image-processor';
import { BLEManager } from '../../utils/ble-manager';

// ============================================================
// Page
// ============================================================
Page({

  data: {
    GRID_SIZE: GRID_SIZE,
    imageSelected: false,
    processing: false,
    hasError: false,
    statusText: '',
    pixelCountText: '',
    pixelArray: null,
    bleScanning: false,
    bleConnected: false,
    bleStatusText: '',
    bleDevices: [],
    connectedDeviceId: '',
    connectedDeviceName: '',
    isSending: false,
    processMode: 'edge-aware',  // 'dither' 照片 | 'edge-aware' 边缘感知(推荐) | 'edge' 徽章
  },

  _displayCanvas: null,
  _displayCtx: null,
  _offscreenCanvas: null,
  _offscreenCtx: null,

  /* ============================================================
     生命周期
     ============================================================ */
  onLoad() {
    console.log('[AI印章] 页面加载');
    this._initBLEManager();
  },

  onReady() {
    this._initAllCanvases();
  },

  onUnload() {
    if (this._bleManager) this._bleManager.close();
  },

  _initBLEManager() {
    const mgr = new BLEManager({
      serviceUUID: BLE_SERVICE_UUID,
      writeCharUUID: BLE_CHAR_UUID_WRITE,
      notifyCharUUID: BLE_CHAR_UUID_NOTIFY,
      deviceNamePrefix: BLE_DEVICE_NAME_PREFIX,
      mtuTarget: MTU_TARGET,
      frameInterval: 30,
      sendTimeout: 5000,
      scanTimeout: 10000,
    });

    mgr.onDeviceFound = (_device, list) => {
      this.setData({
        bleDevices: list,
        bleStatusText: `📡 发现 ${list.length} 台印章设备`,
      });
    };

    mgr.onScanComplete = (list) => {
      this.setData({
        bleScanning: false,
        bleStatusText: list.length > 0 ? '✅ 扫描完成，请选择设备连接' : '⚠️ 未发现印章设备',
      });
    };

    mgr.onConnected = (deviceId) => {
      const name = this.data.bleDevices.find(d => d.deviceId === deviceId)?.name || '印章设备';
      wx.hideLoading();
      this.setData({
        bleConnected: true,
        connectedDeviceId: deviceId,
        connectedDeviceName: name,
        bleStatusText: '✅ 已连接到印章，可发送数据',
      });
      wx.showToast({ title: '连接成功！', icon: 'success' });
    };

    mgr.onDisconnected = () => {
      wx.hideLoading();
      this.setData({
        bleConnected: false,
        connectedDeviceId: '',
        connectedDeviceName: '',
        bleStatusText: '🔌 已断开连接',
        bleDevices: [],
      });
    };

    mgr.onAck = (_data) => {
      wx.hideLoading();
      this.setData({ isSending: false, bleStatusText: '✅ 数据已同步至印章' });
      wx.showToast({ title: '同步成功，请盖章！', icon: 'success', duration: 2000 });
    };

    mgr.onError = (err) => {
      this._showError(typeof err === 'string' ? err : (err.errMsg || err.message || JSON.stringify(err)));
    };

    this._bleManager = mgr;
  },

  /* ============================================================
     初始化 Canvas
     ============================================================ */
  _initAllCanvases() {
    const query = wx.createSelectorQuery();
    query.select('#previewCanvas')
      .fields({ node: true, size: true })
      .exec((res) => {
        if (res && res[0]) {
          this._displayCanvas = res[0].node;
          this._displayCtx = this._displayCanvas.getContext('2d');
          console.log('[AI印章] 显示 Canvas 初始化成功');
        } else {
          console.warn('[AI印章] 显示 Canvas 初始化失败');
        }
      });
    // 离屏 canvas
    try {
      const off = wx.createOffscreenCanvas({ type: '2d', width: 1, height: 1 });
      this._offscreenCanvas = off;
      this._offscreenCtx = off.getContext('2d');
      console.log('[AI印章] 离屏 Canvas 初始化成功');
    } catch (e) {
      console.warn('[AI印章] 离屏 Canvas 创建失败:', e);
    }
  },

  /* ============================================================
     选择图片
     ============================================================ */
  chooseImage() {
    wx.chooseMedia({
      count: 1,
      mediaType: ['image'],
      sourceType: ['album', 'camera'],
      sizeType: ['original'],
      success: (res) => {
        const path = res.tempFiles[0].tempFilePath;
        this.setData({
          imageSelected: true,
          processing: true,
          hasError: false,
          statusText: '⏳ 正在处理图片...',
          pixelCountText: '',
          pixelArray: null,
        });
        this._processImage(path);
      },
      fail: (err) => {
        if (err.errMsg && err.errMsg.indexOf('cancel') === -1) {
          this._showError('选择图片失败：' + JSON.stringify(err));
        }
      },
    });
  },

  /* ============================================================
     ⭐ 图像处理（委托给 image-processor 模块，支持双模式）
     ============================================================ */
  async _processImage(tempFilePath) {
    try {
      const cvs = this._offscreenCanvas || this._displayCanvas;
      const ctx = cvs ? (this._offscreenCtx || this._displayCtx) : null;
      if (!cvs || !ctx) { this._showError('Canvas 尚未初始化'); return; }

      const mode = this.data.processMode || 'dither';
      console.log(`[AI印章] 处理模式: ${mode}`);

      const { pixelArray, blackCount } = await processImage({
        tempFilePath,
        canvas: cvs,
        mode,
        gridSize: GRID_SIZE,
      });

      this.data.pixelArray = pixelArray;
      this.setData({
        pixelCountText: `⚫ 黑色 ${blackCount} 像素  ⚪ 白色 ${TOTAL_PIXELS - blackCount} 像素`,
      });

      this._drawPreview(pixelArray);

      console.log('═══════════════════════════════════════');
      console.log('📦 原始 1024 点阵数组:');
      console.log(pixelArray);
      console.log('───────────────────────────────────────');
      const packedFrame = this._packData(pixelArray);
      console.log('📡 最终 133 字节帧（十六进制）:');
      console.log(this._bytesToHex(packedFrame));
      console.log('═══════════════════════════════════════');

      this.setData({ processing: false, statusText: '✅ 图片处理完成！' });
    } catch (err) {
      console.error('[AI印章] 处理出错:', err);
      this._showError('图像处理失败：' + (err.errMsg || err.message || JSON.stringify(err)));
    }
  },

  /* ============================================================
     切换处理模式
     ============================================================ */
  switchMode(e) {
    const mode = e.currentTarget.dataset.mode;
    const labels = { 'dither': '照片', 'edge-aware': '边缘感知', 'edge': '徽章' };
    this.setData({ processMode: mode });
    if (this.data.imageSelected && this.data.pixelArray) {
      wx.showToast({ title: `已切换为${labels[mode] || mode}模式，请重新选择图片`, icon: 'none', duration: 2000 });
    }
  },

  /* ============================================================
     绘制点阵预览（最近邻插值放大）
     ============================================================ */
  _drawPreview(pixelArray) {
    const cvs = this._displayCanvas;
    const ctx = this._displayCtx;
    if (!cvs || !ctx) { console.warn('[AI印章] Canvas 不可用，跳过预览'); return; }
    const ds = 320;
    const dpr = wx.getSystemInfoSync().pixelRatio;
    cvs.width = ds * dpr;
    cvs.height = ds * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, ds, ds);
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, ds, ds);
    const cs = ds / GRID_SIZE;
    for (let row = 0; row < GRID_SIZE; row++) {
      for (let col = 0; col < GRID_SIZE; col++) {
        ctx.fillStyle = pixelArray[row * GRID_SIZE + col] === 1 ? '#000000' : '#FFFFFF';
        ctx.fillRect(Math.floor(col * cs), Math.floor(row * cs), Math.ceil(cs), Math.ceil(cs));
      }
    }
    console.log('[AI印章] 点阵预览绘制完成');
  },

  /* ============================================================
     ⭐ 数据打包
     ============================================================ */
  _packData(pixelArray) {
    const dataBytes = new Uint8Array(128);
    for (let i = 0; i < TOTAL_PIXELS; i += 8) {
      let byte = 0;
      for (let j = 0; j < 8; j++) {
        if (i + j < TOTAL_PIXELS && pixelArray[i + j] === 1) {
          byte |= (1 << (7 - j));
        }
      }
      dataBytes[i / 8] = byte;
    }
    const frame = new Uint8Array(133);
    frame[0] = SOF;
    frame[1] = CMD;
    frame[2] = 128;
    frame.set(dataBytes, 3);
    let cs = 0;
    for (let i = 0; i < 131; i++) cs = (cs + frame[i]) & 0xFF;
    frame[131] = cs;
    frame[132] = EOF;
    return frame;
  },

  _bytesToHex(bytes) {
    return Array.from(bytes).map(b => b.toString(16).padStart(2, '0').toUpperCase()).join(' ');
  },

  _showError(msg) {
    console.error('[AI印章]', msg);
    this.setData({ processing: false, hasError: true, statusText: '❌ ' + msg });
    wx.showToast({
      title: typeof msg === 'string' && msg.length > 20 ? '操作失败' : msg,
      icon: 'none', duration: 3000,
    });
  },

  /* ============================================================
     蓝牙（委托给 BLEManager）
     ============================================================ */

  startScan() {
    this.setData({ bleScanning: true, bleStatusText: '📡 正在扫描设备...', bleDevices: [], hasError: false });
    this._bleManager.openAdapter()
      .then(() => this._bleManager.startScan())
      .catch((err) => {
        let m = '蓝牙未开启，请先打开手机蓝牙';
        if (err.errCode === 10001) m = '蓝牙未初始化，请检查蓝牙开关';
        else if (err.errCode === 10012) m = '本机蓝牙未开启';
        else if (err.errCode === 10013) m = '请在系统设置中开启蓝牙权限';
        this._showError(m);
        this.setData({ bleScanning: false, bleStatusText: '❌ ' + m });
      });
  },

  stopScan() {
    this._bleManager.stopScan();
  },

  connectDevice(e) {
    const deviceId = e.currentTarget.dataset.deviceid;
    const deviceName = e.currentTarget.dataset.name || '印章设备';
    wx.showLoading({ title: '连接中...' });
    this.setData({ bleStatusText: '🔗 正在连接 ' + deviceName + '...' });
    this._bleManager.connect(deviceId)
      .catch((err) => {
        wx.hideLoading();
        this._showError('连接失败：' + (err.errMsg || err.message || JSON.stringify(err)));
        this.setData({ bleStatusText: '❌ 连接失败' });
      });
  },

  async sendData() {
    const pa = this.data.pixelArray;
    if (!pa || pa.length !== TOTAL_PIXELS) { this._showError('没有可发送的点阵数据，请先选择图片'); return; }
    if (!this.data.bleConnected) { this._showError('请先连接印章设备'); return; }
    this.setData({ isSending: true, bleStatusText: '📤 正在发送点阵数据...' });
    wx.showLoading({ title: '发送中...', mask: true });

    try {
      const frame = this._packData(pa);
      await this._bleManager.sendData(frame);
    } catch (err) {
      wx.hideLoading();
      this.setData({ isSending: false });
      this._showError('发送失败：' + (err.errMsg || err.message || JSON.stringify(err)));
    }
  },

  disconnectDevice() {
    wx.showLoading({ title: '断开中...' });
    this._bleManager.disconnect();
  },

});
