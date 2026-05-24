/* ============================================================
   AI智能印章 - 首页逻辑 (pages/index/index.js)
   图像处理 → 点阵生成 → 数据打包 → 蓝牙通信
   ============================================================ */

const GRID_SIZE = 32;
const TOTAL_PIXELS = GRID_SIZE * GRID_SIZE;

// 蓝牙常量
const BLE_SERVICE_UUID = '0000FFE0-0000-1000-8000-00805F9B34FB';
const BLE_CHAR_UUID_WRITE = '0000FFE1-0000-1000-8000-00805F9B34FB';
const BLE_CHAR_UUID_NOTIFY = '0000FFE2-0000-1000-8000-00805F9B34FB';
const BLE_DEVICE_NAME_PREFIX = 'TIANJI-STAMP-01';
const MTU_TARGET = 247;

// 通信帧协议
const SOF = 0xAA;
const CMD = 0x01;
const EOF = 0xBB;

import { processToDither, processToEdge, processToHalftone, countBlack } from '../../utils/image-processor';
import { BLEManager } from '../../utils/ble-manager';

Page({

  data: {
    GRID_SIZE,
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
    processMode: 'dither',  // 'dither' 照片 | 'edge' 线稿 | 'halftone' 波普
  },

  _displayCanvas: null,
  _displayCtx: null,
  _offscreenCanvas: null,
  _offscreenCtx: null,
  _lastImagePath: '',

  /* ============================================================
     生命周期
     ============================================================ */
  onLoad() {
    this._initBLEManager();
  },

  onReady() {
    this._initAllCanvases();
  },

  onUnload() {
    if (this._bleManager) this._bleManager.close();
    this._releaseOffscreenCanvas();
  },

  /* ============================================================
     Canvas 初始化 & 清理
     ============================================================ */
  _initAllCanvases() {
    // 显示 Canvas（预览用）
    const query = wx.createSelectorQuery();
    query.select('#previewCanvas')
      .fields({ node: true, size: true })
      .exec((res) => {
        if (res && res[0]) {
          this._displayCanvas = res[0].node;
          this._displayCtx = this._displayCanvas.getContext('2d');
        } else {
          console.warn('[AI印章] 显示 Canvas 初始化失败');
        }
      });

    // 离屏 Canvas（图像加载用）
    try {
      const off = wx.createOffscreenCanvas({ type: '2d', width: 1, height: 1 });
      this._offscreenCanvas = off;
      this._offscreenCtx = off.getContext('2d');
    } catch (e) {
      console.warn('[AI印章] 离屏 Canvas 创建失败:', e);
    }
  },

  /** 释放离屏 Canvas 内存：将尺寸重置为 1×1 */
  _releaseOffscreenCanvas() {
    const cvs = this._offscreenCanvas;
    if (cvs) {
      cvs.width = 1;
      cvs.height = 1;
    }
  },

  /* ============================================================
     BLE 初始化（委托给 BLEManager）
     ============================================================ */
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
        bleStatusText: `发现 ${list.length} 台印章设备`,
      });
    };

    mgr.onScanComplete = (list) => {
      this.setData({
        bleScanning: false,
        bleStatusText: list.length > 0 ? '扫描完成，请选择设备连接' : '未发现印章设备',
      });
    };

    mgr.onConnected = (deviceId) => {
      const name = this.data.bleDevices.find(d => d.deviceId === deviceId)?.name || '印章设备';
      wx.hideLoading();
      this.setData({
        bleConnected: true,
        connectedDeviceId: deviceId,
        connectedDeviceName: name,
        bleStatusText: '已连接到印章，可发送数据',
      });
      wx.showToast({ title: '连接成功！', icon: 'success' });
    };

    mgr.onDisconnected = () => {
      wx.hideLoading();
      this.setData({
        bleConnected: false,
        connectedDeviceId: '',
        connectedDeviceName: '',
        bleStatusText: '已断开连接',
        bleDevices: [],
      });
    };

    mgr.onAck = () => {
      wx.hideLoading();
      this.setData({ isSending: false, bleStatusText: '数据已同步至印章' });
      wx.showToast({ title: '同步成功，请盖章！', icon: 'success', duration: 2000 });
    };

    mgr.onError = (err) => {
      this._showError(typeof err === 'string' ? err : (err.errMsg || err.message || JSON.stringify(err)));
    };

    this._bleManager = mgr;
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
          statusText: '正在处理图片...',
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
     图像处理核心：加载图片 → 取 ImageData → 调管线 → 预览
     ============================================================ */
  async _processImage(tempFilePath) {
    const cvs = this._offscreenCanvas;
    if (!cvs) { this._showError('Canvas 尚未初始化'); return; }

    try {
      // --- 1. 加载图片到离屏 Canvas ---
      const img = await new Promise((resolve, reject) => {
        const image = cvs.createImage();
        image.onload = () => resolve(image);
        image.onerror = (e) => reject(e);
        image.src = tempFilePath;
      });

      // --- 2. 绘制并提取像素数据 ---
      cvs.width = img.width;
      cvs.height = img.height;
      const ctx = cvs.getContext('2d');
      ctx.drawImage(img, 0, 0, img.width, img.height);
      const imageData = ctx.getImageData(0, 0, img.width, img.height);

      // --- 3. 调用对应管线 ---
      const mode = this.data.processMode;
      let pixelArray;
      if (mode === 'edge') {
        pixelArray = processToEdge(imageData.data, imageData.width, imageData.height);
      } else if (mode === 'halftone') {
        pixelArray = processToHalftone(imageData.data, imageData.width, imageData.height);
      } else {
        pixelArray = processToDither(imageData.data, imageData.width, imageData.height);
      }

      // --- 4. 释放离屏 Canvas 内存 ---
      this._releaseOffscreenCanvas();

      // --- 5. 更新 UI ---
      const blackCount = countBlack(pixelArray);
      this.data.pixelArray = pixelArray;
      this.setData({
        processing: false,
        statusText: '图片处理完成！',
        pixelCountText: `黑色 ${blackCount} 像素   白色 ${TOTAL_PIXELS - blackCount} 像素`,
      });

      this._drawPreview(pixelArray);

      // --- 6. 控制台日志 ---
      console.log('═══════════════════════════════════════');
      console.log(`管线: ${mode}  |  黑色像素: ${blackCount}`);
      console.log('点阵数组:', Array.from(pixelArray).join(''));
      const frame = this._packData(pixelArray);
      console.log('蓝牙帧 (hex):', this._bytesToHex(frame));
      console.log('═══════════════════════════════════════');

    } catch (err) {
      this._releaseOffscreenCanvas();
      console.error('[AI印章] 处理出错:', err);
      this._showError('图像处理失败：' + (err.errMsg || err.message || JSON.stringify(err)));
    }
  },

  /* ============================================================
     切换处理模式
     ============================================================ */
  switchMode(e) {
    const mode = e.currentTarget.dataset.mode;
    const labels = { dither: '照片复古', edge: '硬朗线稿', halftone: '波普图腾' };
    this.setData({ processMode: mode });
    if (this.data.imageSelected) {
      wx.showToast({
        title: `已切换为【${labels[mode]}】, 请重新选择图片`,
        icon: 'none',
        duration: 2000,
      });
    }
  },

  /* ============================================================
     点阵预览（最近邻插值放大到 320×320 逻辑像素）
     ============================================================ */
  _drawPreview(pixelArray) {
    const cvs = this._displayCanvas;
    const ctx = this._displayCtx;
    if (!cvs || !ctx) return;

    const ds = 320;
    const dpr = wx.getSystemInfoSync().pixelRatio;
    cvs.width = ds * dpr;
    cvs.height = ds * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, ds, ds);

    // 先铺黑底
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, ds, ds);

    const cs = ds / GRID_SIZE;
    for (let row = 0; row < GRID_SIZE; row++) {
      for (let col = 0; col < GRID_SIZE; col++) {
        ctx.fillStyle = pixelArray[row * GRID_SIZE + col] === 1 ? '#000000' : '#FFFFFF';
        ctx.fillRect(Math.floor(col * cs), Math.floor(row * cs), Math.ceil(cs), Math.ceil(cs));
      }
    }
  },

  /* ============================================================
     数据打包：1024 像素 → 133 字节帧
     帧格式：[SOF 1B][CMD 1B][LEN 1B][128B data][CS 1B][EOF 1B]
     ============================================================ */
  _packData(pixelArray) {
    const dataBytes = new Uint8Array(128);
    for (let i = 0; i < TOTAL_PIXELS; i += 8) {
      let byte = 0;
      for (let j = 0; j < 8; j++) {
        if (pixelArray[i + j] === 1) {
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
    this.setData({ processing: false, hasError: true, statusText: msg });
    wx.showToast({
      title: typeof msg === 'string' && msg.length > 20 ? '操作失败' : msg,
      icon: 'none', duration: 3000,
    });
  },

  /* ============================================================
     蓝牙操作（委托给 BLEManager）
     ============================================================ */
  startScan() {
    this.setData({ bleScanning: true, bleStatusText: '正在扫描设备...', bleDevices: [], hasError: false });
    this._bleManager.openAdapter()
      .then(() => this._bleManager.startScan())
      .catch((err) => {
        let m = '蓝牙未开启，请先打开手机蓝牙';
        if (err.errCode === 10001) m = '蓝牙未初始化，请检查蓝牙开关';
        else if (err.errCode === 10012) m = '本机蓝牙未开启';
        else if (err.errCode === 10013) m = '请在系统设置中开启蓝牙权限';
        this._showError(m);
        this.setData({ bleScanning: false, bleStatusText: m });
      });
  },

  stopScan() {
    this._bleManager.stopScan();
  },

  connectDevice(e) {
    const deviceId = e.currentTarget.dataset.deviceid;
    const deviceName = e.currentTarget.dataset.name || '印章设备';
    wx.showLoading({ title: '连接中...' });
    this.setData({ bleStatusText: '正在连接 ' + deviceName + '...' });
    this._bleManager.connect(deviceId)
      .catch((err) => {
        wx.hideLoading();
        this._showError('连接失败：' + (err.errMsg || err.message || JSON.stringify(err)));
        this.setData({ bleStatusText: '连接失败' });
      });
  },

  async sendData() {
    const pa = this.data.pixelArray;
    if (!pa || pa.length !== TOTAL_PIXELS) { this._showError('没有可发送的点阵数据，请先选择图片'); return; }
    if (!this.data.bleConnected) { this._showError('请先连接印章设备'); return; }

    this.setData({ isSending: true, bleStatusText: '正在发送点阵数据...' });
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
