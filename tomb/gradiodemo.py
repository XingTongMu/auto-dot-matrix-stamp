"""
gradiodemo.py —— 图像转黑白线稿 Web 交互界面（Gradio >= 4.0）

本模块使用 Gradio Blocks API 构建完整的 Web 界面，通过调用 test.py 中的
sketch2anime 推理流水线，实现「上传图像 → 参数调节 → 黑白线稿预览」的
一站式交互体验。

使用方式:
    python gradiodemo.py

依赖:
  - test.py（推理流水线 + CLAHE）
  - gradio >= 4.0
"""

import gradio as gr
from PIL import Image
from typing import Optional, Tuple

import test  # 推理流水线模块


# =============================================================================
# 阶段 1：推理包装函数（Gradio 回调）
# =============================================================================
def process_image(
    input_image: Optional[Image.Image],
    mode: str,
    use_clahe: bool,
    clahe_clip: float,
    thickness_scale: float,
    enable_resize: bool,
    output_width: int,
    output_height: int,
) -> Optional[Image.Image]:
    """
    Gradio 回调函数：将前端传入的图像与参数路由至 sketch2anime 推理流水线。

    Args:
        input_image:     用户上传的 PIL Image（由 gr.Image(type="pil") 解析）
        mode:            线稿提取模式 ("default" / "improved")
        use_clahe:       是否启用 CLAHE 局部对比度增强
        clahe_clip:      CLAHE 对比度限制系数
        thickness_scale: 线宽倍率，1.0=基准，2.0=双倍粗，0.5=减半
        enable_resize:   是否启用压缩
        output_width:    输出线稿宽度
        output_height:   输出线稿高度

    Returns:
        黑白线稿 PIL Image。
        若 enable_resize=True，则尺寸为 (output_width, output_height)；
        否则恢复原始分辨率。
    """
    # ---- 空值保护：未上传图片时返回 None ----
    if input_image is None:
        return None

    # ---- 构造 output_size ----
    output_size = None
    if enable_resize and output_width > 0 and output_height > 0:
        output_size = (output_width, output_height)

    # ---- 调用推理流水线 ----
    try:
        result_pil = test.sketch2anime(
            img_obj=input_image,
            mode=mode,
            use_clahe=use_clahe,
            clahe_clip=clahe_clip,
            output_size=output_size,
            thickness_scale=thickness_scale,
        )
        return result_pil
    except Exception as e:
        print(f"[gradiodemo] 推理过程出现异常: {e}")
        raise gr.Error(f"线稿生成失败: {e}")





# =============================================================================
# 阶段 2：构建 Gradio Blocks 界面
# =============================================================================
def build_demo() -> gr.Blocks:
    """
    使用 Gradio Blocks API 构建交互界面。

    界面布局:
      ┌─────────────────────────────────────────────┐
      │  🎨 图像 → 黑白线稿生成器                    │
      │  ┌───────────────────┐  ┌────────────────┐  │
      │  │  📷 上传彩色照片   │  │  🖼 高质量线稿  │  │
      │  │  (拖拽上传区)      │  │  (结果展示区)   │  │
      │  └───────────────────┘  └────────────────┘  │
      │  ⚙️ 参数设置                                │
      │  ○ 提取模式: [default] [improved]            │
      │  ☑ 开启 CLAHE                                │
      │  ──── 对比度限制系数: [2.0] ────             │
      │  [🔄 提交]                                   │
      └─────────────────────────────────────────────┘
    """
    with gr.Blocks(
        title="图像 → 黑白线稿生成器",
    ) as demo:
        # ===================== 标题区域 =====================
        gr.Markdown(
            """
            # 🎨 图像 → 黑白线稿生成器
            ### 上传一张彩色照片，AI 将自动提取线稿，生成高质量黑白简笔画风格图像
            """
        )

        # ===================== 主布局：输入 + 输出 并排 =====================
        with gr.Row(equal_height=True):
            # ---- 左列：输入 ----
            with gr.Column(scale=1):
                input_image = gr.Image(
                    type="pil",
                    label="📷 上传彩色照片",
                    height=400,
                    elem_id="input_image",
                )
                gr.Markdown("*支持 JPG / PNG / BMP 等常见格式*")

            # ---- 右列：输出 ----
            with gr.Column(scale=1):
                output_image = gr.Image(
                    type="pil",
                    label="🖼️ 高质量黑白线稿",
                    height=400,
                    elem_id="output_image",
                    interactive=False,
                )
                gr.Markdown("*线稿保持原始上传照片的分辨率*")

        # ===================== 参数控制区 =====================
        with gr.Accordion("⚙️ 参数设置", open=True):
            with gr.Row():
                # ---- 模式选择 ----
                mode_radio = gr.Radio(
                    choices=["default", "improved"],
                    value="default",
                    label="🎯 提取模式",
                    info="default: 经典 U-Net  |  improved: 平滑上采样（消除棋盘伪影）",
                )

                # ---- CLAHE 开关 ----
                clahe_checkbox = gr.Checkbox(
                    label="开启局部对比度自适应均衡 (CLAHE)",
                    value=False,
                    info="增强局部对比度，防止暗部线条丢失",
                )

            with gr.Row():
                # ---- CLAHE 阈值滑块 ----
                clahe_slider = gr.Slider(
                    minimum=1.0,
                    maximum=5.0,
                    value=2.0,
                    step=0.1,
                    label="🔄 对比度限制系数",
                    info="值越大对比度增强越强（仅在 CLAHE 开启时生效）",
                    interactive=True,
                )

            # ---- CLAHE 开关联动：关闭时禁用滑块 ----
            def toggle_clahe_slider(use_clahe: bool) -> gr.Slider:
                """当 CLAHE 关闭时，滑块置灰不可调"""
                return gr.Slider(interactive=use_clahe)

            clahe_checkbox.change(
                fn=toggle_clahe_slider,
                inputs=clahe_checkbox,
                outputs=clahe_slider,
            )

            # ===================== 线宽倍率调节 =====================
            gr.Markdown("### ✏️ 线宽调节")
            with gr.Row():
                thickness_slider = gr.Slider(
                    minimum=0.25,
                    maximum=5.0,
                    value=1.0,
                    step=0.25,
                    label="线宽倍率",
                    info="1.0=基准线宽，2.0=双倍粗，0.5=减半",
                    interactive=True,
                )

            # ===================== 输出尺寸控制 =====================

            gr.Markdown("### 📐 输出尺寸控制")
            with gr.Row():
                enable_resize = gr.Checkbox(
                    label="压缩线稿到指定尺寸",
                    value=False,
                    info="勾选后将线稿压缩到指定像素尺寸（如 32×32）",
                )
            with gr.Row():
                output_width = gr.Number(
                    value=32,
                    minimum=1,
                    maximum=4096,
                    step=1,
                    label="宽度 (px)",
                    precision=0,
                    interactive=True,
                )
                output_height = gr.Number(
                    value=32,
                    minimum=1,
                    maximum=4096,
                    step=1,
                    label="高度 (px)",
                    precision=0,
                    interactive=True,
                )

            # ---- 尺寸控制开关联动 ----
            def toggle_size_inputs(enabled: bool):
                return gr.Number(interactive=enabled), gr.Number(interactive=enabled)

            enable_resize.change(
                fn=toggle_size_inputs,
                inputs=enable_resize,
                outputs=[output_width, output_height],
            )

        # ===================== 提交按钮 =====================

        with gr.Row():
            submit_btn = gr.Button(
                value="🔄 生成线稿",
                variant="primary",
                scale=2,
            )
            clear_btn = gr.Button(
                value="🗑️ 清空",
                variant="secondary",
                scale=1,
            )

        # ===================== 事件绑定 =====================
        # ---- 点击"生成线稿"按钮 → 执行推理 ----

        submit_btn.click(
            fn=process_image,
            inputs=[input_image, mode_radio, clahe_checkbox, clahe_slider, thickness_slider, enable_resize, output_width, output_height],
            outputs=output_image,
            api_name="sketch2anime",
        )

        # ---- 上传图像时自动触发推理 ----
        input_image.upload(
            fn=process_image,
            inputs=[input_image, mode_radio, clahe_checkbox, clahe_slider, thickness_slider, enable_resize, output_width, output_height],
            outputs=output_image,
        )



        # ---- 清空按钮：重置所有组件 ----
        clear_btn.click(
            fn=lambda: (None, None),  # type: ignore
            inputs=None,
            outputs=[input_image, output_image],
        )


        # ===================== 使用示例 =====================
        gr.Markdown(
            """
            ---
            ### 📖 使用说明
            1. **拖拽或点击上传** 彩色照片到左侧区域
            2. 在 **参数设置** 中选择提取模式和 CLAHE 选项
            3. 点击 **🔄 生成线稿** 按钮或按 Enter 键提交
            4. 右侧展示区将显示 AI 提取的黑白线稿结果
            """
        )

    return demo


# =============================================================================
# 阶段 3：启动入口
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  🎨 图像 → 黑白线稿生成器 (Gradio Web 界面)")
    print("=" * 70)
    print()
    print("  启动中... 请在浏览器中打开显示的 URL")
    print()

    # 构建并启动 Demo
    demo = build_demo()
    demo.launch(
        server_name="0.0.0.0",  # 允许局域网访问
        server_port=7860,        # Gradio 默认端口
        share=False,             # 不生成公网链接（改为 True 可创建临时公网链接）
        show_error=True,         # 在前端显示错误信息
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1100px;
            margin: auto;
        }
        """,
    )
