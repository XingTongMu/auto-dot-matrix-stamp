/* ============================================================
   AI智能印章 - 蓝牙通信模块 (utils/ble-manager.js)
   功能：扫描/连接/MTU协商/发送队列(防死锁)/ACK超时重试
   ============================================================ */

// ============================================================
// BLEManager 类
// ============================================================
class BLEManager {
  constructor(options = {}) {
    this._serviceUUID = options.serviceUUID || '';
    this._writeCharUUID = options.writeCharUUID || '';
    this._notifyCharUUID = options.notifyCharUUID || '';
    this._deviceNamePrefix = options.deviceNamePrefix || '';
    this._mtuTarget = options.mtuTarget || 247;
    this._frameInterval = options.frameInterval || 30;    // 帧间延迟 ms
    this._sendTimeout = options.sendTimeout || 5000;      // 单帧 ACK 超时 ms
    this._scanTimeout = options.scanTimeout || 10000;     // 扫描超时 ms

    // 内部状态
    this._deviceId = null;
    this._serviceId = null;
    this._writeCharId = null;
    this._notifyCharId = null;
    this._connected = false;
    this._scanning = false;
    this._devices = [];

    // 发送队列
    this._sendQueue = [];
    this._sending = false;
    this._sendTimer = null;
    this._currentResolve = null;
    this._currentReject = null;
    this._scanTimer = null;

    // 回调
    this.onDeviceFound = null;    // (device, deviceList) => void
    this.onScanComplete = null;   // (deviceList) => void
    this.onConnected = null;      // (deviceId) => void
    this.onDisconnected = null;   // () => void
    this.onAck = null;            // (data) => void
    this.onError = null;          // (err) => void

    // 绑定 this
    this._handleDeviceFound = this._handleDeviceFound.bind(this);
    this._handleNotify = this._handleNotify.bind(this);
  }

  // ============================================================
  // 扫描
  // ============================================================

  async openAdapter() {
    return new Promise((resolve, reject) => {
      wx.openBluetoothAdapter({
        mode: 'central',
        success: () => { console.log('[BLE] 适配器打开成功'); resolve(); },
        fail: (err) => reject(err),
      });
    });
  }

  startScan() {
    this._devices = [];
    this._scanning = true;
    wx.onBluetoothDeviceFound(this._handleDeviceFound);

    return new Promise((resolve, reject) => {
      wx.startBluetoothDevicesDiscovery({
        allowDuplicatesKey: false,
        interval: 0,
        success: () => {
          console.log('[BLE] 开始扫描');
          this._scanTimer = setTimeout(() => {
            if (this._scanning) this.stopScan();
          }, this._scanTimeout);
          resolve();
        },
        fail: (err) => {
          this._scanning = false;
          wx.offBluetoothDeviceFound(this._handleDeviceFound);
          reject(err);
        },
      });
    });
  }

  stopScan() {
    if (this._scanTimer) { clearTimeout(this._scanTimer); this._scanTimer = null; }
    if (!this._scanning) return;
    this._scanning = false;
    wx.stopBluetoothDevicesDiscovery({
      complete: () => {
        wx.offBluetoothDeviceFound(this._handleDeviceFound);
        if (this.onScanComplete) this.onScanComplete([...this._devices]);
      },
    });
  }

  _handleDeviceFound(res) {
    const devices = res.devices || [];
    for (const d of devices) {
      if (d.name && d.name.indexOf(this._deviceNamePrefix) !== -1) {
        if (!this._devices.some(x => x.deviceId === d.deviceId)) {
          const device = { deviceId: d.deviceId, name: d.name, RSSI: d.RSSI || -100 };
          this._devices.push(device);
          if (this.onDeviceFound) this.onDeviceFound(device, [...this._devices]);
        }
      }
    }
  }

  // ============================================================
  // 连接
  // ============================================================

  async connect(deviceId) {
    this.stopScan();
    this._deviceId = deviceId;

    await this._createConnection(deviceId);
    await this._negotiateMTU(deviceId);
    await this._discoverServices(deviceId);
    await this._discoverCharacteristics(deviceId);

    this._connected = true;
    if (this.onConnected) this.onConnected(deviceId);
    return deviceId;
  }

  _createConnection(deviceId) {
    return new Promise((resolve, reject) => {
      wx.createBLEConnection({
        deviceId,
        timeout: 10000,
        success: () => resolve(),
        fail: (err) => {
          this._deviceId = null;
          reject(err);
        },
      });
    });
  }

  _negotiateMTU(deviceId) {
    return new Promise((resolve) => {
      wx.setBLEMTU({
        deviceId,
        mtu: this._mtuTarget,
        success: () => { console.log('[BLE] MTU 协商成功:', this._mtuTarget); resolve(); },
        fail: (err) => { console.warn('[BLE] MTU 协商失败:', err); resolve(); },
      });
    });
  }

  _discoverServices(deviceId) {
    return new Promise((resolve, reject) => {
      wx.getBLEDeviceServices({
        deviceId,
        success: (res) => {
          const svc = res.services.find(
            s => s.uuid.toUpperCase() === this._serviceUUID.toUpperCase()
          );
          if (svc) {
            this._serviceId = svc.uuid;
            resolve();
          } else {
            reject(new Error('未找到印章服务 UUID'));
          }
        },
        fail: (err) => reject(err),
      });
    });
  }

  _discoverCharacteristics(deviceId) {
    return new Promise((resolve, reject) => {
      wx.getBLEDeviceCharacteristics({
        deviceId,
        serviceId: this._serviceId,
        success: (res) => {
          const writeChar = res.characteristics.find(
            c => c.uuid.toUpperCase() === this._writeCharUUID.toUpperCase()
          );
          const notifyChar = res.characteristics.find(
            c => c.uuid.toUpperCase() === this._notifyCharUUID.toUpperCase()
          );
          if (!writeChar) {
            reject(new Error('未找到写入特征 UUID'));
            return;
          }
          this._writeCharId = writeChar.uuid;
          if (notifyChar) {
            this._notifyCharId = notifyChar.uuid;
            this._enableNotify(deviceId);
          }
          resolve();
        },
        fail: (err) => reject(err),
      });
    });
  }

  _enableNotify(deviceId) {
    wx.notifyBLECharacteristicValueChange({
      deviceId,
      serviceId: this._serviceId,
      characteristicId: this._notifyCharId,
      state: true,
      success: () => {
        console.log('[BLE] 通知开启成功');
        wx.offBLECharacteristicValueChange(this._handleNotify);
        wx.onBLECharacteristicValueChange(this._handleNotify);
      },
      fail: (err) => { console.warn('[BLE] 开启通知失败:', err); },
    });
  }

  _handleNotify(res) {
    if (!this._notifyCharId) return;
    if (res.characteristicId.toUpperCase() !== this._notifyCharId.toUpperCase()) return;

    console.log('[BLE] 收到硬件 ACK');
    if (this._sendTimer) { clearTimeout(this._sendTimer); this._sendTimer = null; }

    const resolve = this._currentResolve;
    this._currentResolve = null;
    this._currentReject = null;
    this._sending = false;

    if (this.onAck) this.onAck(res.value);

    // ACK 成功后 resolve 当前帧，延迟后处理下一帧
    if (resolve) resolve();
    setTimeout(() => this._processQueue(), this._frameInterval);
  }

  // ============================================================
  // 发送队列（防死锁核心）
  //
  // 架构：
  //   sendData() → _sendQueue.push()
  //              → _processQueue() 串行弹出
  //              → 单包直接写 / 大包分包写
  //              → 写完成后等待 notify ACK（_handleNotify）
  //              → ACK 到达 → resolve() → 延迟 → 处理下一帧
  //              → 超时 → reject() → 处理下一帧
  // ============================================================

  /**
   * 发送一帧数据，返回 Promise
   * @param {Uint8Array} frame - 完整帧（含帧头帧尾校验）
   * @returns {Promise<void>} 收到硬件 ACK 时 resolve，超时 reject
   */
  sendData(frame) {
    return new Promise((resolve, reject) => {
      this._sendQueue.push({ frame, resolve, reject });
      if (!this._sending) this._processQueue();
    });
  }

  _processQueue() {
    if (this._sendQueue.length === 0) {
      this._sending = false;
      return;
    }

    this._sending = true;
    const { frame, resolve, reject } = this._sendQueue.shift();
    this._currentResolve = resolve;
    this._currentReject = reject;

    // 单帧超时定时器
    this._sendTimer = setTimeout(() => {
      console.warn('[BLE] 单帧 ACK 超时');
      this._sendTimer = null;
      const r = this._currentReject;
      this._currentResolve = null;
      this._currentReject = null;
      this._sending = false;
      if (r) r(new Error('BLE ACK 超时'));
      setTimeout(() => this._processQueue(), this._frameInterval);
    }, this._sendTimeout);

    // 根据 MTU 选择发送策略
    if (this._mtuTarget >= frame.length) {
      this._writeChunk(frame, 0, frame.length);
    } else {
      this._sendFragmented(frame, frame.length, 20);
    }
  }

  /**
   * 单包写入
   */
  _writeChunk(data, offset, length) {
    const chunk = data.slice(offset, offset + length);
    wx.writeBLECharacteristicValue({
      deviceId: this._deviceId,
      serviceId: this._serviceId,
      characteristicId: this._writeCharId,
      value: chunk.buffer,
      success: () => {
        console.log(`[BLE] 写入成功: ${chunk.length}B`);
        // 不 resolve — 等待硬件 ACK (notify)
      },
      fail: (err) => {
        console.error('[BLE] 写入失败:', err);
        this._abortSend(err);
      },
    });
  }

  /**
   * 分包写入（MTU 不足时使用）
   */
  _sendFragmented(frame, totalLength, fragSize) {
    let offset = 0;
    const next = () => {
      if (offset >= totalLength) {
        console.log('[BLE] 分包发送完成，等待 ACK');
        return; // 全部发完，等待 notify ACK
      }
      const end = Math.min(offset + fragSize, totalLength);
      const chunk = frame.slice(offset, end);
      wx.writeBLECharacteristicValue({
        deviceId: this._deviceId,
        serviceId: this._serviceId,
        characteristicId: this._writeCharId,
        value: chunk.buffer,
        success: () => {
          console.log(`[BLE] 分包: ${offset}-${end}/${totalLength}`);
          offset = end;
          setTimeout(next, this._frameInterval);
        },
        fail: (err) => {
          console.error('[BLE] 分包写入失败:', err);
          this._abortSend(err);
        },
      });
    };
    next();
  }

  /**
   * 异常中断当前帧发送
   */
  _abortSend(err) {
    if (this._sendTimer) { clearTimeout(this._sendTimer); this._sendTimer = null; }
    const r = this._currentReject;
    this._currentResolve = null;
    this._currentReject = null;
    this._sending = false;
    if (r) r(err);
    setTimeout(() => this._processQueue(), this._frameInterval);
  }

  // ============================================================
  // 断开 / 清理
  // ============================================================

  get isConnected() {
    return this._connected;
  }

  get devices() {
    return [...this._devices];
  }

  get connectedDeviceId() {
    return this._deviceId;
  }

  async disconnect() {
    // 清空发送队列
    this._abortAllPending();
    wx.offBLECharacteristicValueChange(this._handleNotify);

    if (!this._deviceId) return;

    return new Promise((resolve) => {
      wx.closeBLEConnection({
        deviceId: this._deviceId,
        complete: () => {
          this._connected = false;
          this._deviceId = null;
          this._serviceId = null;
          this._writeCharId = null;
          this._notifyCharId = null;
          if (this.onDisconnected) this.onDisconnected();
          resolve();
        },
      });
    });
  }

  _abortAllPending() {
    if (this._sendTimer) { clearTimeout(this._sendTimer); this._sendTimer = null; }
    const reject = this._currentReject;
    this._currentResolve = null;
    this._currentReject = null;
    this._sending = false;
    if (reject) reject(new Error('连接已断开'));

    while (this._sendQueue.length > 0) {
      const item = this._sendQueue.shift();
      item.reject(new Error('连接已断开'));
    }
  }

  close() {
    this.stopScan();
    this._abortAllPending();
    wx.offBLECharacteristicValueChange(this._handleNotify);
    wx.closeBluetoothAdapter({});
  }
}

export { BLEManager };
