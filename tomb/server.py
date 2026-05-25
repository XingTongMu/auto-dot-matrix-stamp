"""
server.py —— 线稿生成 API 后端服务 (Flask)

提供 HTTP API 接口，供微信小程序前端调用。
同时保留后续接硬件设备的扩展能力。

使用方式:
    python server.py

API 接口:
    POST /api/sketch2anime
        上传图片 → 返回线稿图片（二进制 PNG）
    GET  /api/health
        健康检查
"""

import io
import os
import sys
import base64
import uuid
import traceback
from datetime import datetime

from flask import Flask, request, jsonify, send_file
from PIL import Image
from werkzeug.utils import secure_filename

from typing import Optional, Tuple

import test  # 推理流水线模块


# =============================================================================
# Flask 应用初始化
# =============================================================================
app = Flask(__name__)

# 手动 CORS 支持：所有路由允许跨域
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
    response.headers["Access-Control-Max-Age"] = "86400"
    return response

# 配置文件
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 最大上传 16MB
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "temp_uploads")
app.config["RESULT_FOLDER"] = os.path.join(os.path.dirname(__file__), "temp_results")

# 确保临时目录存在
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["RESULT_FOLDER"], exist_ok=True)

# 支持的图片格式
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}

def allowed_file(filename: str) -> bool:
    """检查文件扩展名是否允许"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# =============================================================================
# API 路由
# =============================================================================
@app.route("/api/health", methods=["GET"])
def health_check():
    """健康检查接口"""
    return jsonify({
        "status": "ok",
        "service": "sketch2anime",
        "timestamp": datetime.now().isoformat(),
    })


def _parse_output_size(size_str: Optional[str]) -> Optional[Tuple[int, int]]:
    """
    解析 output_size 字符串参数。
    格式: "32,32" → (32, 32)
    空字符串或 None → None（保持原始分辨率）
    """
    if not size_str:
        return None
    try:
        parts = size_str.split(",")
        w, h = int(parts[0].strip()), int(parts[1].strip())
        if w <= 0 or h <= 0:
            return None
        return (w, h)
    except (ValueError, IndexError):
        return None


@app.route("/api/sketch2anime_upload", methods=["POST"])
def sketch2anime_upload_api():
    """
    微信小程序 uploadFile 专用 API：接受 multipart 上传，返回 JSON + base64

    请求方式: POST
    请求体 (multipart/form-data):
        - image: 图片文件（必须）
        - mode: 提取模式（默认 "default"）
        - use_clahe: "true"/"false"（默认 "false"）
        - clahe_clip: 对比度系数（默认 2.0）
        - output_size: 输出尺寸，格式 "width,height"，如 "32,32"（可选）
        - thickness_scale: 线宽倍率，如 "1.5"（可选，默认 1.0）

    返回:
        {
            "success": true/false,
            "image_base64": "base64编码的线稿PNG",
            "width": 640,
            "height": 480,
            "error": "错误信息（失败时）"
        }
    """
    # ---- 1. 检查文件 ----
    if "image" not in request.files:
        return jsonify({"success": False, "error": "未找到上传的图片文件"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "error": "未选择图片文件"}), 400

    # ---- 2. 读取参数 ----
    mode = request.form.get("mode", "default")
    use_clahe_str = request.form.get("use_clahe", "false")
    clahe_clip_str = request.form.get("clahe_clip", "2.0")
    output_size_str = request.form.get("output_size", "")
    thickness_scale_str = request.form.get("thickness_scale", "1.0")
    use_clahe = use_clahe_str.lower() == "true"
    try:
        clahe_clip = float(clahe_clip_str)
    except ValueError:
        clahe_clip = 2.0
    try:
        thickness_scale = float(thickness_scale_str)
    except ValueError:
        thickness_scale = 1.0
    output_size = _parse_output_size(output_size_str)


    if mode not in ("default", "improved"):
        return jsonify({"success": False, "error": f"不支持的 mode='{mode}'"}), 400

    # ---- 3. 读取图片 ----
    try:
        image_data = file.read()
        pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception as e:
        return jsonify({"success": False, "error": f"图片解析失败: {str(e)}"}), 400

    # ---- 4. 执行推理 ----
    try:
        result_pil = test.sketch2anime(
            img_obj=pil_image,
            mode=mode,
            use_clahe=use_clahe,
            clahe_clip=clahe_clip,
            output_size=output_size,
            thickness_scale=thickness_scale,
        )
        # ---- 5. 编码为 Base64 PNG ----

        img_buffer = io.BytesIO()
        result_pil.save(img_buffer, format="PNG")
        img_buffer.seek(0)
        result_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
        return jsonify({
            "success": True,
            "image_base64": result_base64,
            "width": result_pil.width,
            "height": result_pil.height,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500



@app.route("/api/sketch2anime", methods=["POST"])
def sketch2anime_api():
    """
    核心 API：图片 → 黑白线稿

    请求方式: POST
    请求体 (multipart/form-data):
        - image: 图片文件（必须）
        - mode: 提取模式，可选 "default" 或 "improved"（默认 "default"）
        - use_clahe: 是否启用 CLAHE，可选 "true" 或 "false"（默认 "false"）
        - clahe_clip: CLAHE 对比度限制系数（默认 2.0）
        - output_size: 输出尺寸，格式 "width,height"，如 "32,32"（可选）
        - thickness_scale: 线宽倍率，如 "1.5"（可选，默认 1.0）

    返回:
        - 成功: 200, image/png（线稿图片二进制数据）
        - 失败: 400/500, application/json（错误信息）
    """
    # ---- 1. 检查文件是否存在 ----
    if "image" not in request.files:
        return jsonify({"error": "未找到上传的图片文件，请使用字段名 'image'"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "未选择图片文件"}), 400

    # ---- 2. 读取参数 ----
    mode = request.form.get("mode", "default")
    use_clahe_str = request.form.get("use_clahe", "false")
    clahe_clip_str = request.form.get("clahe_clip", "2.0")
    output_size_str = request.form.get("output_size", "")
    thickness_scale_str = request.form.get("thickness_scale", "1.0")

    use_clahe = use_clahe_str.lower() == "true"
    try:
        clahe_clip = float(clahe_clip_str)
    except ValueError:
        clahe_clip = 2.0
    try:
        thickness_scale = float(thickness_scale_str)
    except ValueError:
        thickness_scale = 1.0
    output_size = _parse_output_size(output_size_str)

    # ---- 3. 验证模式 ----
    if mode not in ("default", "improved"):
        return jsonify({"error": f"不支持的 mode='{mode}'，可选: 'default', 'improved'"}), 400

    # ---- 4. 读取图片 ----
    try:
        image_data = file.read()
        pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"图片解析失败: {str(e)}"}), 400

    # ---- 5. 执行推理 ----
    try:
        result_pil = test.sketch2anime(
            img_obj=pil_image,
            mode=mode,
            use_clahe=use_clahe,
            clahe_clip=clahe_clip,
            output_size=output_size,
            thickness_scale=thickness_scale,
        )



        # ---- 6. 将结果编码为 PNG 二进制流 ----
        img_buffer = io.BytesIO()
        result_pil.save(img_buffer, format="PNG")
        img_buffer.seek(0)

        # ---- 7. 返回图片 ----
        return send_file(
            img_buffer,
            mimetype="image/png",
            as_attachment=False,
        )

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": f"线稿生成失败: {str(e)}",
            "detail": traceback.format_exc(),
        }), 500


@app.route("/api/sketch2anime_b64", methods=["POST", "OPTIONS"])
def sketch2anime_b64_api():
    # OPTIONS 预检请求直接返回成功
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
    """
    Base64 版本 API（适合某些硬件设备的调用方式）

    请求方式: POST
    请求体 (application/json):
        {
            "image_base64": "base64编码的图片数据",
            "mode": "default",
            "use_clahe": false,
            "clahe_clip": 2.0,
            "output_size": "32,32"  （可选，格式 "width,height"）
        }

    返回:
        {
            "success": true,
            "image_base64": "base64编码的线稿PNG",
            "width": 640,
            "height": 480
        }
    """
    data = request.get_json(force=True)
    if not data or "image_base64" not in data:
        return jsonify({"error": "缺少 image_base64 字段"}), 400

    # ---- 解码 Base64 图片 ----
    try:
        base64_str = data["image_base64"]
        # 兼容 data:image/png;base64, 前缀
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        image_bytes = base64.b64decode(base64_str)
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Base64 图片解码失败: {str(e)}"}), 400

    # ---- 读取参数 ----
    mode = data.get("mode", "default")
    use_clahe = data.get("use_clahe", False)
    clahe_clip = data.get("clahe_clip", 2.0)
    output_size_str = data.get("output_size", "")
    thickness_scale = data.get("thickness_scale", 1.0)
    output_size = _parse_output_size(output_size_str)

    # ---- 执行推理 ----
    try:
        result_pil = test.sketch2anime(
            img_obj=pil_image,
            mode=mode,
            use_clahe=use_clahe,
            clahe_clip=clahe_clip,
            output_size=output_size,
            thickness_scale=thickness_scale,
        )



        # ---- 编码结果为 Base64 PNG ----
        img_buffer = io.BytesIO()
        result_pil.save(img_buffer, format="PNG")
        result_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")

        return jsonify({
            "success": True,
            "image_base64": result_base64,
            "width": result_pil.width,
            "height": result_pil.height,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e),
        }), 500


# =============================================================================
# 启动入口
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  🎨 线稿生成 API 服务 (Flask)")
    print("=" * 70)
    print()
    print("  API 端点:")
    print("    POST /api/sketch2anime      - 上传图片，返回线稿 PNG")
    print("    POST /api/sketch2anime_b64  - Base64 版本 API")
    print("    GET  /api/health            - 健康检查")
    print()
    print("  启动地址: http://localhost:5001")
    print()

    app.run(
        host="0.0.0.0",
        port=5001,
        debug=False,
    )
