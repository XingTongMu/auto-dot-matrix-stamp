"""
model.py —— 线稿提取核心网络模块

包含两种架构 + 统一路由接口：
  1. UnetGenerator (default)  — 经典 U-Net，8 层下采样 / 8 层上采样，跳跃连接
  2. ImprovedUpsampleSmooth (improved)
     — 用「双线性插值 + 高斯平滑 + MLP 残差微调」替代转置卷积，消除棋盘伪影
  3. create_model(mode)       — 统一调用路由
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple


# =============================================================================
# 常量定义：编码器各层输出通道数（共 8 层）
# =============================================================================
ENCODER_CHANNELS: List[int] = [64, 128, 256, 512, 512, 512, 512, 512]
"""编码器 8 层输出通道:  3→64→128→256→512→512→512→512→512"""

DECODER_CHANNELS: List[int] = [512, 512, 512, 256, 128, 64, 32]
"""解码器前 7 层输出通道（第 8 层输出 1 通道 + Tanh）"""


# =============================================================================
# 阶段 1：基础构建块
# =============================================================================
class EncoderBlock(nn.Module):
    """
    编码器块：Conv2d(stride=2) → BatchNorm → LeakyReLU
    每个块将特征图空间尺寸减半，通道数翻倍（或保持不变）。
    """
    def __init__(self, in_ch: int, out_ch: int, use_bn: bool = True):
        """
        Args:
            in_ch:  输入通道数
            out_ch: 输出通道数
            use_bn: 是否使用 BatchNorm（第一层和最后一层可关闭）
        """
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入 [B, C, H, W] → 输出 [B, out_ch, H/2, W/2]"""
        return self.lrelu(self.bn(self.conv(x)))


class DecoderBlock(nn.Module):
    """
    解码器块：上采样 →（可选跳跃连接）→ Conv2d → BN → Dropout → ReLU
    输入的通道数 = up_ch + skip_ch（若有跳跃连接）
    """
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        """
        Args:
            in_ch:  输入通道数（已拼接跳跃连接后的总通道数）
            out_ch: 输出通道数
            dropout: Dropout2d 概率（前 3 层高 dropout 防过拟合）
        """
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入 [B, in_ch, H, W] → 输出 [B, out_ch, H, W]"""
        return self.relu(self.dropout(self.bn(self.conv(x))))


# =============================================================================
# 阶段 2：平滑上采样组件（ImprovedUpsampleSmooth 用）
# =============================================================================
class GaussianSmoothing(nn.Module):
    """
    固定 3×3 高斯平滑滤波器（不可学习参数）。
    核权重初始化为二维高斯分布：中间高、四周低。
    通过 F.conv2d(..., groups=channels) 逐通道应用。
    """
    def __init__(self, channels: int):
        """
        Args:
            channels: 输入特征图的通道数（逐通道独立平滑）
        """
        super().__init__()
        # ---- 2a. 构建 3×3 高斯核（二项式近似） ----
        #     1  2  1
        #     2  4  2  / 16
        #     1  2  1
        kernel = torch.tensor([[1.0, 2.0, 1.0],
                               [2.0, 4.0, 2.0],
                               [1.0, 2.0, 1.0]]) / 16.0
        # 扩展为 [C, 1, 3, 3]，每个通道一个独立核
        kernel = kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        # 注册为持久化缓冲区（非可训练参数）
        self.register_buffer("gaussian_kernel", kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] 特征图
        Returns:
            [B, C, H, W] 高斯平滑后的特征图
        """
        # ---- 2b. 使用 ReflectionPad 保持边缘一致性 ----
        x_padded = F.pad(x, (1, 1, 1, 1), mode="reflect")
        # ---- 2c. 逐通道分组卷积 ----
        # groups=C 确保每个通道独立平滑
        return F.conv2d(x_padded, self.gaussian_kernel, groups=x.shape[1])


class MLPResidualBlock(nn.Module):
    """
    MLP 残差微调块：Conv2d(1×1) → GELU → Conv2d(1×1)
    输出: mlp(x) + x（残差连接）
    """
    def __init__(self, channels: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """残差连接: output = MLP(x) + x"""
        mlp_out = self.mlp(x)
        return mlp_out + x


class SmoothUpsampleBlock(nn.Module):
    """
    平滑上采样块（ImprovedUpsampleSmooth 核心组件）:
      1. F.interpolate(scale_factor=2, mode='bilinear') 放大 2 倍
      2. GaussianSmoothing → 3×3 高斯平滑滤波
      3. MLPResidualBlock → 1×1 Conv → GELU → 1×1 Conv + 残差
      4. 拼接跳跃连接特征（若有）
      5. Conv2d(3×3) → BN → ReLU 调整通道数
    """
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        """
        Args:
            in_ch:  上采样前特征图的通道数
            out_ch: 输出通道数
            dropout: Dropout 概率
        """
        super().__init__()
        # ---- 平滑上采样管线 ----
        self.smooth_pipeline = nn.Sequential(
            GaussianSmoothing(in_ch),
            MLPResidualBlock(in_ch),
        )
        # ---- 拼接后卷积 ----
        # 拼接跳跃连接后，输入通道数 = in_ch + in_ch（因为跳跃连接来自对应编码层）
        # 但实际上对应编码层输出通道为 out_ch*2，需要根据具体传入调整
        # 使用灵活设计：init 时不固定 skip_ch，在 forward 接收 skip
        self.conv = nn.Conv2d(in_ch * 2, out_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x:    来自解码器上一层的特征图 [B, in_ch, H, W]
            skip: 编码器对应层的跳跃连接 [B, in_ch, H*2, W*2]（上采样后尺寸匹配）
        Returns:
            [B, out_ch, H*2, W*2] 特征图
        """
        # ---- 步骤 1: 双线性插值上采样 2 倍 ----
        x_up = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)

        # ---- 步骤 2: 高斯平滑 + MLP 残差微调 ----
        x_smooth = self.smooth_pipeline(x_up)

        # ---- 步骤 3: 拼接跳跃连接（编码器底层几何特征） ----
        if skip is not None:
            # 确保空间尺寸一致
            if x_smooth.shape[2:] != skip.shape[2:]:
                x_smooth = F.interpolate(
                    x_smooth, size=skip.shape[2:], mode="bilinear", align_corners=False
                )
            x_cat = torch.cat([x_smooth, skip], dim=1)
        else:
            # 无跳跃连接时，用零填充保持通道数一致
            x_cat = torch.cat([x_smooth, torch.zeros_like(x_smooth)], dim=1)

        # ---- 步骤 4: 卷积调整通道 + BN + Dropout + ReLU ----
        x_out = self.relu(self.dropout(self.bn(self.conv(x_cat))))
        return x_out


# =============================================================================
# 阶段 3：完整网络架构
# =============================================================================
class UnetGenerator(nn.Module):
    """
    经典 U-Net 生成器（default 模式）。

    架构概览:
      编码器: 8× EncoderBlock (Conv2d stride=2 + BN + LeakyReLU)
              特征图:  512→256→128→64→32→16→8→4→2
              通道数:    3→64→128→256→512→512→512→512→512
      解码器: 8× DecoderBlock (Bilinear ↑2 + Cat Skip + Conv + BN + Dropout + ReLU)
              最终输出: Conv2d(3×3) + Tanh → [B, 1, H, W]

    Forward 过程完整保留 7 层跳跃连接（编码器 d1~d7 → 解码器对应层）。
    """
    def __init__(self):
        super().__init__()

        # ===================== 编码器（下采样） =====================
        # 8 层步进卷积: stride=2 每次将空间尺寸减半
        # d1: 第一层不使用 BN
        self.d1 = EncoderBlock(3, 64, use_bn=False)
        self.d2 = EncoderBlock(64, 128)
        self.d3 = EncoderBlock(128, 256)
        self.d4 = EncoderBlock(256, 512)
        self.d5 = EncoderBlock(512, 512)
        self.d6 = EncoderBlock(512, 512)
        self.d7 = EncoderBlock(512, 512)
        # d8: 最后一层不使用 BN（ bottleneck ）
        self.d8 = EncoderBlock(512, 512, use_bn=False)

        # ===================== 解码器（上采样） =====================
        # 共 8 层上采样，前 3 层使用 Dropout=0.5 防过拟合
        self.u1 = DecoderBlock(512 + 512, 512, dropout=0.5)   # d7 skip
        self.u2 = DecoderBlock(512 + 512, 512, dropout=0.5)   # d6 skip
        self.u3 = DecoderBlock(512 + 512, 512, dropout=0.5)   # d5 skip
        self.u4 = DecoderBlock(512 + 512, 256)                 # d4 skip
        self.u5 = DecoderBlock(256 + 256, 128)                 # d3 skip
        self.u6 = DecoderBlock(128 + 128, 64)                  # d2 skip
        self.u7 = DecoderBlock(64 + 64, 32)                    # d1 skip

        # ===================== 输出层 =====================
        # 上采样 2 倍 → Conv(32→1, 3×3) → Tanh
        self.out_conv = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Tanh(),
        )

        # ---- 初始化权重（Mock 兼容） ----
        self._init_weights()

    def _init_weights(self):
        """使用均值为 0，标准差为 0.02 的正态分布初始化所有 Conv2d 权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, mean=1.0, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入图像 [B, 3, H, W]，像素值域 [-1.0, 1.0]
        Returns:
            线稿图 [B, 1, H, W]，像素值域 [-1.0, 1.0]（Tanh 输出）
        """
        # ===================== 编码器前向 =====================
        # 记录每一层输出用于跳跃连接（d1 ~ d7）
        s1 = self.d1(x)   # [B,  64, H/2,   W/2]
        s2 = self.d2(s1)  # [B, 128, H/4,   W/4]
        s3 = self.d3(s2)  # [B, 256, H/8,   W/8]
        s4 = self.d4(s3)  # [B, 512, H/16,  W/16]
        s5 = self.d5(s4)  # [B, 512, H/32,  W/32]
        s6 = self.d6(s5)  # [B, 512, H/64,  W/64]
        s7 = self.d7(s6)  # [B, 512, H/128, W/128]
        s8 = self.d8(s7)  # [B, 512, H/256, W/256]  ← Bottleneck

        # ===================== 解码器前向 =====================
        # ---- 第 1 层: 上采样 + d7 跳跃连接 ----
        x = F.interpolate(s8, scale_factor=2, mode="bilinear", align_corners=False)
        x = torch.cat([x, s7], dim=1)  # [B, 1024, H/128, W/128]
        x = self.u1(x)                 # [B, 512, H/128, W/128]

        # ---- 第 2 层: 上采样 + d6 跳跃连接 ----
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = torch.cat([x, s6], dim=1)  # [B, 1024, H/64, W/64]
        x = self.u2(x)                 # [B, 512, H/64, W/64]

        # ---- 第 3 层: 上采样 + d5 跳跃连接 ----
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = torch.cat([x, s5], dim=1)  # [B, 1024, H/32, W/32]
        x = self.u3(x)                 # [B, 512, H/32, W/32]

        # ---- 第 4 层: 上采样 + d4 跳跃连接 ----
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = torch.cat([x, s4], dim=1)  # [B, 1024, H/16, W/16]
        x = self.u4(x)                 # [B, 256, H/16, W/16]

        # ---- 第 5 层: 上采样 + d3 跳跃连接 ----
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = torch.cat([x, s3], dim=1)  # [B, 512, H/8, W/8]
        x = self.u5(x)                 # [B, 128, H/8, W/8]

        # ---- 第 6 层: 上采样 + d2 跳跃连接 ----
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = torch.cat([x, s2], dim=1)  # [B, 256, H/4, W/4]
        x = self.u6(x)                 # [B, 64, H/4, W/4]

        # ---- 第 7 层: 上采样 + d1 跳跃连接 ----
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = torch.cat([x, s1], dim=1)  # [B, 128, H/2, W/2]
        x = self.u7(x)                 # [B, 32, H/2, W/2]

        # ---- 第 8 层（输出）: 上采样至原尺寸 + Tanh ----
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.out_conv(x)           # [B, 1, H, W]

        return x


class ImprovedUpsampleSmooth(nn.Module):
    """
    改进型平滑上采样 U-Net（improved 模式）。

    与 UnetGenerator 的区别：
      - 解码器中的上采样使用 SmoothUpsampleBlock 替代普通双线性插值
      - SmoothUpsampleBlock 包含：插值 ↑2 → 高斯平滑 → MLP 残差微调 → cat skip → Conv

    消除棋盘状伪影（Checkerboard Artifacts），生成更干净的线稿。
    """
    def __init__(self):
        super().__init__()

        # ===================== 编码器（与 UnetGenerator 相同） =====================
        self.d1 = EncoderBlock(3, 64, use_bn=False)
        self.d2 = EncoderBlock(64, 128)
        self.d3 = EncoderBlock(128, 256)
        self.d4 = EncoderBlock(256, 512)
        self.d5 = EncoderBlock(512, 512)
        self.d6 = EncoderBlock(512, 512)
        self.d7 = EncoderBlock(512, 512)
        self.d8 = EncoderBlock(512, 512, use_bn=False)

        # ===================== 解码器（使用 SmoothUpsampleBlock） =====================
        # 每个 SmoothUpsampleBlock 内部包含: 插值↑2 → 高斯平滑 → MLP残差 → cat skip → Conv
        self.u1 = SmoothUpsampleBlock(512, 512, dropout=0.5)
        self.u2 = SmoothUpsampleBlock(512, 512, dropout=0.5)
        self.u3 = SmoothUpsampleBlock(512, 512, dropout=0.5)
        self.u4 = SmoothUpsampleBlock(512, 256)
        self.u5 = SmoothUpsampleBlock(256, 128)
        self.u6 = SmoothUpsampleBlock(128, 64)
        self.u7 = SmoothUpsampleBlock(64, 32)

        # ===================== 输出层 =====================
        self.out_conv = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Tanh(),
        )

        # ---- 初始化权重 ----
        self._init_weights()

    def _init_weights(self):
        """均值为 0，标准差 0.02 的正态分布初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, mean=1.0, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入图像 [B, 3, H, W]，值域 [-1.0, 1.0]
        Returns:
            线稿图 [B, 1, H, W]，值域 [-1.0, 1.0]
        """
        # ===================== 编码器 =====================
        s1 = self.d1(x)   # [B,  64, H/2,   W/2]
        s2 = self.d2(s1)  # [B, 128, H/4,   W/4]
        s3 = self.d3(s2)  # [B, 256, H/8,   W/8]
        s4 = self.d4(s3)  # [B, 512, H/16,  W/16]
        s5 = self.d5(s4)  # [B, 512, H/32,  W/32]
        s6 = self.d6(s5)  # [B, 512, H/64,  W/64]
        s7 = self.d7(s6)  # [B, 512, H/128, W/128]
        s8 = self.d8(s7)  # [B, 512, H/256, W/256]  ← Bottleneck

        # ===================== 解码器（平滑上采样） =====================
        x = self.u1(s8, s7)   # [B, 512, H/128, W/128]
        x = self.u2(x, s6)    # [B, 512, H/64,  W/64]
        x = self.u3(x, s5)    # [B, 512, H/32,  W/32]
        x = self.u4(x, s4)    # [B, 256, H/16,  W/16]
        x = self.u5(x, s3)    # [B, 128, H/8,   W/8]
        x = self.u6(x, s2)    # [B,  64, H/4,   W/4]
        x = self.u7(x, s1)    # [B,  32, H/2,   W/2]

        # ===================== 输出 =====================
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.out_conv(x)   # [B, 1, H, W]

        return x


# =============================================================================
# 阶段 4：统一路由接口
# =============================================================================
def create_model(mode: str = "default") -> nn.Module:
    """
    根据模式创建对应的线稿提取网络。

    Args:
        mode: 模型模式
            - "default"  : 返回 UnetGenerator（经典 U-Net）
            - "improved" : 返回 ImprovedUpsampleSmooth（平滑上采样，消除棋盘伪影）

    Returns:
        nn.Module: 线稿生成网络

    Raises:
        ValueError: 不支持的 mode 值
    """
    if mode == "default":
        return UnetGenerator()
    elif mode == "improved":
        return ImprovedUpsampleSmooth()
    else:
        raise ValueError(
            f"不支持的 mode='{mode}'，可选值: 'default', 'improved'"
        )


# =============================================================================
# 模块自测
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("model.py 模块自测")
    print("=" * 70)

    # ---- 测试 1: UnetGenerator (default 模式) ----
    print("\n[测试 1] 创建 UnetGenerator (default 模式)")
    model_default = create_model("default")
    total_params = sum(p.numel() for p in model_default.parameters())
    trainable_params = sum(p.numel() for p in model_default.parameters() if p.requires_grad)
    print(f"       参数量: 总计 {total_params:,} | 可训练 {trainable_params:,}")

    # ---- 模拟前向传播 ----
    dummy_input = torch.randn(2, 3, 512, 512)  # [B=2, C=3, H=512, W=512]
    with torch.no_grad():
        output = model_default(dummy_input)
    print(f"       输入形状: {dummy_input.shape} → 输出形状: {output.shape}")
    assert output.shape == (2, 1, 512, 512), f"输出形状错误: {output.shape}"
    print(f"       输出值域: [{output.min():.4f}, {output.max():.4f}]")
    assert output.min() >= -1.0 and output.max() <= 1.0, "Tanh 输出应在 [-1, 1]"
    print("       ✓ 通过")

    # ---- 测试 2: ImprovedUpsampleSmooth (improved 模式) ----
    print("\n[测试 2] 创建 ImprovedUpsampleSmooth (improved 模式)")
    model_improved = create_model("improved")
    total_params_i = sum(p.numel() for p in model_improved.parameters())
    print(f"       参数量: 总计 {total_params_i:,}")

    with torch.no_grad():
        output_i = model_improved(dummy_input)
    print(f"       输入形状: {dummy_input.shape} → 输出形状: {output_i.shape}")
    assert output_i.shape == (2, 1, 512, 512), f"输出形状错误: {output_i.shape}"
    print(f"       输出值域: [{output_i.min():.4f}, {output_i.max():.4f}]")
    assert output_i.min() >= -1.0 and output_i.max() <= 1.0, "Tanh 输出应在 [-1, 1]"
    print("       ✓ 通过")

    # ---- 测试 3: 高斯平滑组件 ----
    print("\n[测试 3] GaussianSmoothing 组件测试")
    smoother = GaussianSmoothing(channels=3)
    test_feat = torch.randn(1, 3, 8, 8)
    smooth_out = smoother(test_feat)
    assert smooth_out.shape == test_feat.shape, f"高斯平滑形状不匹配: {smooth_out.shape}"
    print(f"       输入形状: {test_feat.shape} → 输出形状: {smooth_out.shape}")
    print("       ✓ 通过")

    # ---- 测试 4: MLP 残差块 ----
    print("\n[测试 4] MLPResidualBlock 组件测试")
    mlp_block = MLPResidualBlock(channels=64)
    test_mlp = torch.randn(1, 64, 16, 16)
    mlp_out = mlp_block(test_mlp)
    assert mlp_out.shape == test_mlp.shape, f"MLP 残差形状不匹配: {mlp_out.shape}"
    # 验证残差连接: output ≠ input
    assert not torch.allclose(mlp_out, test_mlp), "MLP 残差块未改变特征"
    print(f"       输入形状: {test_mlp.shape} → 输出形状: {mlp_out.shape}")
    print("       ✓ 通过")

    # ---- 测试 5: 不同空间尺寸兼容性 ----
    print("\n[测试 5] 不同输入尺寸兼容性")
    for size in [256, 512, 1024]:
        inp = torch.randn(1, 3, size, size)
        with torch.no_grad():
            out_default = model_default(inp)
            out_improved = model_improved(inp)
        assert out_default.shape == (1, 1, size, size), f"Default 模式 {size}x{size} 失败: {out_default.shape}"
        assert out_improved.shape == (1, 1, size, size), f"Improved 模式 {size}x{size} 失败: {out_improved.shape}"
        print(f"       {size}×{size}: default ✓ | improved ✓")

    print("\n[测试 6] create_model 异常路径测试")
    try:
        create_model("unknown")
        assert False, "应抛出 ValueError"
    except ValueError as e:
        print(f"       ✓ 正确捕获异常: {e}")

    print("\n" + "=" * 70)
    print("所有测试通过 ✓")
    print("=" * 70)
