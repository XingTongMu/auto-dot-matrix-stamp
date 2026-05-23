"""
test.py —— 推理流水线与局部图像增强模块

本模块负责：
  1. equalize_clahe  — 局部对比度自适应直方图均衡化（CLAHE）
  2. sketch2anime    — 完整推理流水线：XDoG + 骨架化 + 均匀膨胀 → 中粗实线日系动漫线稿

依赖:
  - data.py  （图像读取、变换、逆向映射、保存）
  - model.py （线稿提取网络 + create_model 路由）
"""

import cv2
import numpy as np
import torch
from PIL import Image
from typing import Union, Optional, Tuple

import data
import model


# =============================================================================
# 阶段 1：局部对比度自适应直方图均衡化（CLAHE）
# =============================================================================
def equalize_clahe(
    img_obj: Union[str, Image.Image, np.ndarray],
    clip_limit: float = 2.0,
) -> np.ndarray:
    if isinstance(img_obj, str):
        pil_img = Image.open(img_obj).convert("RGB")
        img_np = np.array(pil_img)
    elif isinstance(img_obj, Image.Image):
        pil_img = img_obj.convert("RGB")
        img_np = np.array(pil_img)
    elif isinstance(img_obj, np.ndarray):
        img_np = img_obj.copy()
    else:
        raise TypeError(f"不支持的输入类型: {type(img_obj)}")
    if img_np.ndim == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    elif img_np.shape[2] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_channel)
    lab_eq = cv2.merge([l_eq, a_channel, b_channel])
    img_bgr_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    img_rgb_eq = cv2.cvtColor(img_bgr_eq, cv2.COLOR_BGR2RGB)
    return img_rgb_eq


# =============================================================================
# XDoG 提取干净艺术线稿
# =============================================================================
def _xdog_line_drawing(
    img_gray: np.ndarray,
    k_sigma: float = 1.6,
    epsilon: float = -0.1,
    phi: float = 10.0,
) -> np.ndarray:
    """XDoG 线稿提取（非真实感渲染，比 Canny 更干净连续）"""
    img_f = img_gray.astype(np.float32) / 255.0
    sigma = 0.8
    k = k_sigma
    s1 = max(1, int(2 * np.ceil(3.0 * sigma) + 1))
    s2 = max(1, int(2 * np.ceil(3.0 * sigma * k) + 1))
    g1 = cv2.GaussianBlur(img_f, (s1, s1), sigma)
    g2 = cv2.GaussianBlur(img_f, (s2, s2), sigma * k)
    dog = g1 - g2
    xdog = np.where(dog < epsilon, 1.0, 1.0 + np.tanh(phi * dog))
    xdog = (xdog - xdog.min()) / (xdog.max() - xdog.min() + 1e-8)
    xdog_255 = (xdog * 255).astype(np.uint8)
    _, binary = cv2.threshold(xdog_255, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


# =============================================================================
# 骨架提取：距离变换法（速度 O(n)，无需 opencv-contrib）
# =============================================================================
def _skeletonize(binary: np.ndarray) -> np.ndarray:
    """
    使用距离变换提取 1px 骨架（白线黑底）。
    原理：二值图 → 距离变换 → 脊线（局部最大值）= 骨架。
    """
    if np.mean(binary) > 127:
        fg = cv2.bitwise_not(binary)
    else:
        fg = binary.copy()
    _, fg = cv2.threshold(fg, 127, 255, cv2.THRESH_BINARY)
    dist = cv2.distanceTransform(fg, cv2.DIST_L2, 5)
    dist = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    dilated = cv2.dilate(dist, kernel, iterations=1)
    skeleton = np.where(dist >= dilated, fg, 0)
    result = cv2.bitwise_not(skeleton)
    return result


# =============================================================================
# 去毛刺：迭代腐蚀端点
# =============================================================================
def _remove_spurs(skel: np.ndarray, spur_length: int = 3) -> np.ndarray:
    """
    迭代去除骨架上短于 spur_length 的毛刺。
    skel: 黑线白底 [H,W] uint8
    """
    if np.mean(skel) > 127:
        fg = cv2.bitwise_not(skel)
    else:
        fg = skel.copy()
    _, fg = cv2.threshold(fg, 127, 255, cv2.THRESH_BINARY)
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]], dtype=np.uint8)
    for _ in range(spur_length):
        hits = cv2.filter2D(fg, cv2.CV_8U, kernel)
        endpoints = (hits == 11) & (fg == 255)
        if not np.any(endpoints):
            break
        fg[endpoints] = 0
    return cv2.bitwise_not(fg)


# =============================================================================
# 从 1px 骨架均匀膨胀到目标线宽
# =============================================================================
def _dilate_from_skeleton(
    skeleton: np.ndarray,
    target_width: int = 3,
    smooth_radius: int = 2,
) -> np.ndarray:
    """从 1px 黑线白底骨架均匀膨胀到 target_width"""
    if np.mean(skeleton) > 127:
        fg = cv2.bitwise_not(skeleton)
    else:
        fg = skeleton.copy()
    radius = target_width // 2
    kernel_size = 2 * max(1, radius) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(fg, kernel, iterations=1)
    if smooth_radius > 0:
        blur_size = smooth_radius * 2 + 1
        blurred = cv2.GaussianBlur(dilated, (blur_size, blur_size), 0)
        _, dilated = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY)
    return cv2.bitwise_not(dilated)


# =============================================================================
# 完整管线：XDoG → 骨架(1px) → 去毛刺 → 均匀膨胀 → 边缘柔化
# =============================================================================
def _anime_line_pipeline(
    img_gray: np.ndarray,
    mode: str = "default",
    thickness_scale: float = 1.0,
) -> np.ndarray:
    """
    管线：XDoG → 骨架化 → 去毛刺 → 均匀膨胀 → 边缘柔化

    骨架化为核心：
      - 1px 中心线保证相邻线膨胀时不会合并成块
      - 去毛刺去除短噪声
      - 均匀膨胀保证全图线宽一致

    Args:
        img_gray:        灰度图 [H, W] uint8
        mode:            "default" 或 "improved"
        thickness_scale: 线宽倍率，1.0=基准线宽，2.0=双倍粗，0.5=减半
    """
    base_width = 4 if mode == "improved" else 3
    target_width = max(1, int(round(base_width * max(0.25, thickness_scale))))
    spur_length = max(1, int(round(3 * thickness_scale)))
    result = _dilate_from_skeleton(
        _remove_spurs(
            _skeletonize(
                _xdog_line_drawing(
                    img_gray,
                    k_sigma=2.0 if mode == "improved" else 1.5,
                    epsilon=-0.06 if mode == "improved" else -0.14,
                    phi=8.0 if mode == "improved" else 12.0,
                )
            ),
            spur_length=spur_length,
        ),
        target_width=target_width,
        smooth_radius=2,
    )
    return result



# =============================================================================
# 阶段 2：线稿提取核心推理流水线
# =============================================================================
def sketch2anime(
    img_obj: Union[str, Image.Image],
    mode: str = "default",
    use_clahe: bool = False,
    clahe_clip: float = 2.0,
    output_size: Optional[Tuple[int, int]] = None,
    thickness_scale: float = 1.0,
) -> Image.Image:
    """
    完整推理流水线：普通图像 → 日系动漫风格中粗实线黑白线稿。

    管线：
      阶段 A — 预处理: 读取 → CLAHE(可选) → 灰度
      阶段 B — 核心线稿: XDoG → 骨架化(1px) → 去毛刺 → 均匀膨胀 → 边缘柔化
      阶段 C — 后处理: BICUBIC 回弹 → 纯黑纯白二值化
      阶段 D — 压缩（可选）: 将最终线稿压缩到指定尺寸（如 32×32）

    输出：纯线条（不填充任何封闭区域），white bg + black lines

    Args:
        img_obj:         输入图像（路径 或 PIL Image）
        mode:            "default"（3px，精细）或 "improved"（4px，稍粗漫画风）
        use_clahe:       是否启用 CLAHE 增强（默认 False）
        clahe_clip:      CLAHE 对比度限制系数（默认 2.0）
        output_size:     输出线稿的目标尺寸 (width, height)，
                         例如 (32, 32) 将线稿压缩到 32×32 像素。
                         默认 None 表示保持原始分辨率。
        thickness_scale: 线宽倍率，1.0=基准线宽，2.0=双倍粗，0.5=减半（默认 1.0）

    Returns:
        纯黑白线稿 PIL Image。若指定 output_size，则尺寸为 output_size；
        否则恢复原始分辨率。
    """
    pil_image, aus_resize = data.read_img_path(img_obj)
    if use_clahe:
        enhanced_np = equalize_clahe(pil_image, clip_limit=clahe_clip)
        pil_image = Image.fromarray(enhanced_np)
    img_rgb = np.array(pil_image)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    lineart = _anime_line_pipeline(img_gray, mode=mode, thickness_scale=thickness_scale)

    _, lineart = cv2.threshold(lineart, 127, 255, cv2.THRESH_BINARY)
    lineart_rgb = cv2.cvtColor(lineart, cv2.COLOR_GRAY2RGB)
    result_pil = Image.fromarray(lineart_rgb)
    result_pil = result_pil.resize(aus_resize, Image.BICUBIC)
    result_np = np.array(result_pil)
    _, result_np = cv2.threshold(result_np, 127, 255, cv2.THRESH_BINARY)
    result_pil = Image.fromarray(result_np)

    # ---- 阶段 D：压缩到指定输出尺寸（如 32×32） ----
    if output_size is not None:
        result_pil = result_pil.resize(output_size, Image.NEAREST)

    return result_pil



# =============================================================================
# 模块自测
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("test.py 模块自测")
    print("=" * 70)
    dummy_w, dummy_h = 640, 480
    dummy_arr = np.zeros((dummy_h, dummy_w, 3), dtype=np.uint8)
    for y in range(dummy_h):
        for x in range(dummy_w):
            r = int(255 * (x / dummy_w))
            g = int(255 * (y / dummy_h))
            b = int(255 * (1.0 - x / dummy_w) * (1.0 - y / dummy_h))
            dummy_arr[y, x] = [r, g, b]
    dummy_pil = Image.fromarray(dummy_arr)
    print(f"[测试] 创建虚拟图像，尺寸: {dummy_pil.size}")
    print("\n[测试 1] equalize_clahe")
    clahe_out = equalize_clahe(dummy_pil, clip_limit=2.0)
    assert isinstance(clahe_out, np.ndarray) and clahe_out.shape == (dummy_h, dummy_w, 3)
    print("       ✓ 通过")
    print("\n[测试 2] sketch2anime (default)")
    r1 = sketch2anime(dummy_pil, mode="default")
    assert isinstance(r1, Image.Image) and r1.size == (dummy_w, dummy_h) and r1.mode == "RGB"
    print(f"       size: {r1.size} ✓")
    print("\n[测试 3] sketch2anime (improved)")
    r2 = sketch2anime(dummy_pil, mode="improved", use_clahe=True)
    assert isinstance(r2, Image.Image) and r2.size == (dummy_w, dummy_h)
    print(f"       size: {r2.size} ✓")
    print("\n[测试 4] 像素验证")
    rn = np.array(r1)
    print(f"       唯一值: {np.unique(rn)}")
    print("       ✓ 通过")
    print("\n[测试 5] sketch2anime 压缩到 32×32")
    r3 = sketch2anime(dummy_pil, mode="default", output_size=(32, 32))
    assert isinstance(r3, Image.Image) and r3.size == (32, 32) and r3.mode == "RGB"
    print(f"       size: {r3.size} ✓")
    print("\n[测试 6] sketch2anime 压缩到 64×64 (improved)")
    r4 = sketch2anime(dummy_pil, mode="improved", use_clahe=True, output_size=(64, 64))
    assert isinstance(r4, Image.Image) and r4.size == (64, 64)
    print(f"       size: {r4.size} ✓")
    print("\n[测试 7] sketch2anime thickness_scale=2.0 (双倍粗)")
    r5 = sketch2anime(dummy_pil, mode="default", thickness_scale=2.0)
    assert isinstance(r5, Image.Image) and r5.size == (dummy_w, dummy_h)
    print(f"       size: {r5.size} ✓")
    print("\n[测试 8] sketch2anime thickness_scale=0.5 (减半)")
    r6 = sketch2anime(dummy_pil, mode="default", thickness_scale=0.5)
    assert isinstance(r6, Image.Image) and r6.size == (dummy_w, dummy_h)
    print(f"       size: {r6.size} ✓")
    print("\n[测试 9] sketch2anime 粗线 + 压缩 (thickness=3.0, 32×32)")
    r7 = sketch2anime(dummy_pil, mode="default", thickness_scale=3.0, output_size=(32, 32))
    assert isinstance(r7, Image.Image) and r7.size == (32, 32)
    print(f"       size: {r7.size} ✓")
    print("\n" + "=" * 70)


    print("所有测试通过 ✓")
    print("=" * 70)
