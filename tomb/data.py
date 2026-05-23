"""
data.py —— 图像输入、转换、逆向映射与无损回弹模块

本模块负责：
  1. 读取图片并记录原始尺寸 (read_img_path)
  2. 构建标准化预处理管线 (get_transform)
  3. 将网络输出的张量逆向映射回 NumPy 图像 (tensor_to_img)
  4. 将图像无损回弹至原始分辨率并保存 (save_image)
"""

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from typing import Tuple, Union, Optional


# =============================================================================
# 阶段 1：图像读取与原始尺寸记录
# =============================================================================
def read_img_path(img_path: Union[str, Image.Image]) -> Tuple[Image.Image, Tuple[int, int]]:
    """
    读取图片，记录原始尺寸 (width, height) 作为 aus_resize，
    强制转换为 RGB 三通道模式，消除 Alpha / CMYK 等非 RGB 通道。

    Args:
        img_path: 图片路径（str）或 PIL Image 对象

    Returns:
        (PIL Image 对象, (原始宽度, 原始高度))
    """
    # ---- 1a. 类型兼容：支持 str 路径或 PIL Image 对象 ----
    if isinstance(img_path, str):
        pil_image = Image.open(img_path)
    elif isinstance(img_path, Image.Image):
        pil_image = img_path.copy()
    else:
        raise TypeError(f"img_path 参数类型不支持，期望 str 或 PIL.Image，实际为 {type(img_path)}")

    # ---- 1b. 记录原始图片尺寸 (width, height) 作为 aus_resize ----
    aus_resize = pil_image.size  # (width, height)

    # ---- 1c. 强制转换至 RGB 三通道 ----
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    return pil_image, aus_resize


# =============================================================================
# 阶段 2：图像预处理管线（标准化 Transform）
# =============================================================================
def get_transform(target_size: int = 512) -> transforms.Compose:
    """
    构建 torchvision.transforms 组合管线：
      1. Resize: 双三次插值 (Bicubic) → (target_size, target_size)
      2. ToTensor: PIL → [C, H, W] 张量，像素值缩放至 [0.0, 1.0]
      3. Normalize: 均值 0.5 / 标准差 0.5，映射像素值至 [-1.0, 1.0]

    Args:
        target_size: 目标正方形边长，默认 512

    Returns:
        torchvision.transforms.Compose 对象
    """
    transform_pipeline = transforms.Compose([
        # ---- 2a. 双三次插值 Resize ----
        transforms.Resize((target_size, target_size), interpolation=transforms.InterpolationMode.BICUBIC),
        # ---- 2b. PIL → Tensor [C, H, W], 值域 [0.0, 1.0] ----
        transforms.ToTensor(),
        # ---- 2c. 标准化至 [-1.0, 1.0] ── 公式: (x - 0.5) / 0.5 ----
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return transform_pipeline


# =============================================================================
# 阶段 3：网络输出张量 → NumPy 图像（逆向映射）
# =============================================================================
def tensor_to_img(image_tensor: torch.Tensor) -> np.ndarray:
    """
    将网络输出的 4 维张量 [B, C, H, W] 逆向映射为 [H, W, C] 的 NumPy 图像。

    执行步骤：
      1. 去批次：取 image_tensor[0]，[B, C, H, W] → [C, H, W]
      2. 逆归一化：像素值从 [-1.0, 1.0] 映射回 [0.0, 255.0]
      3. 类型转换：强制转换为 np.uint8
      4. 通道广播：若 C == 1，沿通道轴复制 3 份 → [3, H, W]
      5. 轴转置：[C, H, W] → [H, W, C]

    Args:
        image_tensor: 形状为 [B, C, H, W] 的 PyTorch 张量

    Returns:
        形状为 [H, W, C] 的 NumPy uint8 矩阵
    """
    # ---- 3a. 去批次：取第一个样本，[B, C, H, W] → [C, H, W] ----
    # 同时分离计算图，detach 后移至 CPU
    if image_tensor.dim() == 4:
        image_tensor = image_tensor[0]          # [B, C, H, W] → [C, H, W]
    elif image_tensor.dim() == 3:
        pass  # 已为 [C, H, W]，无需处理
    else:
        raise ValueError(f"输入张量维度不合法，期望 3 或 4 维，实际为 {image_tensor.dim()} 维")

    # ---- 3b. 逆归一化：[-1.0, 1.0] → [0.0, 255.0] ----
    # 公式: (Input + 1.0) / 2.0 * 255.0
    image_tensor = (image_tensor.detach().cpu() + 1.0) / 2.0 * 255.0

    # ---- 3c. 转换为 NumPy，强制 uint8 ----
    numpy_img = image_tensor.numpy().astype(np.uint8)  # 此时形状仍为 [C, H, W]

    # ---- 3d. 通道广播：若 C == 1，复制为 3 通道 ----
    if numpy_img.shape[0] == 1:
        # np.tile: 沿通道轴复制 3 次，[1, H, W] → [3, H, W]
        numpy_img = np.tile(numpy_img, (3, 1, 1))

    # ---- 3e. 轴转置：[C, H, W] → [H, W, C] ----
    # 符合 NumPy / PIL 规范
    numpy_img = np.transpose(numpy_img, (1, 2, 0))

    return numpy_img


# =============================================================================
# 阶段 4：图像保存与无损回弹（恢复原始分辨率）
# =============================================================================
def save_image(
    numpy_img: np.ndarray,
    aus_resize: Tuple[int, int],
    save_path: Optional[str] = None,
) -> Image.Image:
    """
    将 [H, W, C] NumPy 图像保存为文件，并通过反向双三次插值无损回弹至原始尺寸。

    Args:
        numpy_img:  形状为 [H, W, C] 的 NumPy uint8 矩阵
        aus_resize: 原始图片尺寸 (width, height)
        save_path:  可选的文件保存路径

    Returns:
        恢复原始分辨率后的 PIL Image 对象
    """
    # ---- 4a. NumPy → PIL Image ----
    pil_image = Image.fromarray(numpy_img)

    # ---- 4b. 无损回弹至原始尺寸（反向双三次插值） ----
    # 将图像拉伸放大恢复至 aus_resize 记录的原始尺寸
    pil_image = pil_image.resize(aus_resize, resample=Image.BICUBIC)

    # ---- 4c. 可选：保存至磁盘 ----
    if save_path is not None:
        pil_image.save(save_path)

    return pil_image


# =============================================================================
# 模块自测（仅在直接运行时执行）
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("data.py 模块自测")
    print("=" * 60)

    # ---- 生成一张虚拟测试图 ----
    dummy_img = Image.new("RGB", (640, 480), color=(128, 128, 128))
    print(f"[测试] 创建虚拟图片，原始尺寸: {dummy_img.size}")

    # ---- 测试 read_img_path ----
    img, aus_resize = read_img_path(dummy_img)
    print(f"[测试] read_img_path → 模式: {img.mode}, aus_resize: {aus_resize}")

    # ---- 测试 get_transform ----
    transform = get_transform(target_size=512)
    print(f"[测试] get_transform → Transform 管线构建完成")

    # ---- 测试完整前向数据流 ----
    input_tensor = transform(img)                # [C, H, W]
    batch_tensor = input_tensor.unsqueeze(0)     # [1, C, H, W] 模拟批次
    print(f"[测试] transform 后张量形状: {batch_tensor.shape}, 值域范围: [{batch_tensor.min():.2f}, {batch_tensor.max():.2f}]")

    # ---- 模拟网络输出（Dummy） -------
    # 模拟一个线稿生成网络输出（单通道线稿风格）
    mock_output = torch.randn(1, 1, 512, 512)   # [B=1, C=1, H=512, W=512]
    print(f"[测试] 模拟网络输出形状: {mock_output.shape}")

    # ---- 测试 tensor_to_img ----
    recovered_np = tensor_to_img(mock_output)
    print(f"[测试] tensor_to_img → NumPy 形状: {recovered_np.shape}, dtype: {recovered_np.dtype}")

    # ---- 测试 save_image ----
    final_img = save_image(recovered_np, aus_resize)
    print(f"[测试] save_image → 最终尺寸: {final_img.size}, 模式: {final_img.mode}")

    # ---- 断言验证 ----
    assert final_img.size == aus_resize, f"尺寸不匹配: {final_img.size} != {aus_resize}"
    assert recovered_np.shape[2] == 3, f"通道数不是 3: {recovered_np.shape[2]}"
    assert recovered_np.dtype == np.uint8, f"dtype 不是 uint8: {recovered_np.dtype}"

    print(f"[测试] 全部断言通过 ✓")
    print("=" * 60)
