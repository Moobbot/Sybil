import os
import uuid
import zipfile
import shutil
from flask import Blueprint, request, jsonify, send_file, send_from_directory
from config import RESULTS_FOLDER, UPLOAD_FOLDER
from call_model import load_model, predict
from utils import (
    create_zip_result,
    dicom_to_png,
    extract_zip_file,
    get_overlay_files,
    get_valid_files,
    save_uploaded_files,
    get_file_path,
    save_uploaded_zip,
)

bp = Blueprint("routes", __name__)

model = load_model()


@bp.route("/api_predict", methods=["POST"])
def api_predict():
    """API để nhận file ZIP, giải nén, chạy mô hình, và trả về file ZIP kết quả"""

    print("API predict_zip called")
    file = request.files.get("file")

    if not file or file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not file.filename.endswith(".zip"):
        return jsonify({"error": "Invalid file format. Only ZIP is allowed."}), 400

    print("File upload:", file)

    # Tạo UUID cho mỗi yêu cầu dự đoán
    session_id = str(uuid.uuid4())

    # Lưu file ZIP tải lên
    zip_path = save_uploaded_zip(file, session_id, folder_save=UPLOAD_FOLDER)

    # Giải nén file ZIP
    unzip_path, error_response, status_code = extract_zip_file(
        zip_path, session_id, folder_save=UPLOAD_FOLDER
    )
    if error_response:
        return error_response, status_code

    # Kiểm tra xem có file hợp lệ
    valid_files = get_valid_files(unzip_path)

    if not valid_files:
        shutil.rmtree(unzip_path)  # Xóa thư mục trống
        return jsonify({"error": "No valid files found in the ZIP archive"}), 400

    # Thư mục để lưu kết quả dự đoán
    output_dir = os.path.join(RESULTS_FOLDER, session_id)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Session ID: {session_id}, Output directory: {output_dir}")

    # Chạy dự đoán
    pred_dict, _ = predict(
        unzip_path, output_dir, model, visualize_attentions_img=True, save_as_dicom=True
    )

    overlay_images_link = os.path.join(output_dir, "serie_0")
    # Nén kết quả dự đoán
    result_zip_path = create_zip_result(
        overlay_images_link, session_id, folder_save=RESULTS_FOLDER
    )

    # Xóa thư mục trung gian
    shutil.rmtree(unzip_path)
    shutil.rmtree(output_dir)

    base_url = request.host_url.rstrip("/")
    zip_download_link = f"{base_url}/download_zip/{session_id}"

    # Trả về kết quả JSON bao gồm link tải file ZIP
    response = {
        "session_id": session_id,
        "predictions": pred_dict["predictions"],
        "overlay_images": zip_download_link,
        "message": "Prediction successful.",
    }

    return jsonify(response)


@bp.route("/api_predict_v1", methods=["POST"])
def api_predict_v1():
    """API to receive photos, run the model, and return the prediction"""

    print("API predict called")
    files = request.files.getlist("file")

    if not files or all(file.filename == "" for file in files):
        return jsonify({"error": "No selected files"}), 400

    # Tạo UUID cho mỗi yêu cầu dự đoán
    session_id = str(uuid.uuid4())

    # Lưu file & lấy danh sách tệp đã tải lên
    uploaded_files, upload_path = save_uploaded_files(files, session_id)
    if not uploaded_files:
        return jsonify({"error": "No valid files uploaded"}), 400

    output_dir = os.path.join(RESULTS_FOLDER, session_id)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Session ID: {session_id}, Output directory: {output_dir}")

    # Chạy dự đoán
    pred_dict, overlayed_images = predict(
        upload_path,
        output_dir,
        model,
        visualize_attentions_img=True,
        save_as_dicom=True,
    )

    # Lấy danh sách file overlay
    overlay_files = get_overlay_files(output_dir, session_id)
    base_url = request.host_url.rstrip("/")
    overlay_image_info = [
        {
            "filename": img,
            "download_link": f"{base_url}/download/{session_id}/{img}",
            "preview_link": f"{base_url}/preview/{session_id}/{img}",
        }
        for img in overlay_files
    ]

    # Trả về kết quả JSON bao gồm đường dẫn và attention values
    response = {
        "session_id": session_id,
        "predictions": pred_dict["predictions"],
        "overlay_images": overlay_image_info,
        "gif_download": (
            f"{base_url}/download_gif/{session_id}" if overlay_files else None
        ),
        "message": "Prediction successful.",
    }

    return jsonify(response)


@bp.route("/download/<session_id>/<filename>", methods=["GET"])
def download_file(session_id, filename):
    """API to download Overlay image according to Session ID."""
    file_path = get_file_path(session_id, filename)

    if os.path.exists(file_path):
        print(f"✅ File found: {file_path}, preparing download...")
        return send_file(file_path, as_attachment=True)

    print(f"⚠️ File not found: {file_path}")
    return (
        jsonify(
            {"error": "File not found", "session_id": session_id, "filename": filename}
        ),
        404,
    )


@bp.route("/download_zip/<session_id>", methods=["GET"])
def download_zip(session_id):
    """API to download Overlay image according to Session ID."""
    file_path = os.path.join(RESULTS_FOLDER, session_id + ".zip")
    if os.path.exists(file_path):
        print(f"✅ File found: {file_path}, preparing download...")
        return send_file(file_path, as_attachment=True)

    print(f"⚠️ File not found: {file_path}")
    return (
        jsonify({"error": "File not found", "session_id": session_id}),
        404,
    )


@bp.route("/preview/<session_id>/<filename>", methods=["GET"])
def preview_file(session_id, filename):
    """API to preview overlay photos"""
    overlay_dir = os.path.join(RESULTS_FOLDER, session_id, "serie_0")
    file_path = os.path.join(overlay_dir, filename)

    if os.path.exists(file_path):
        print(f"✅ Previewing file: {file_path}")
        return send_from_directory(overlay_dir, filename)

    print(f"⚠️ Preview file not found: {file_path}")
    return (
        jsonify(
            {"error": "File not found", "session_id": session_id, "filename": filename}
        ),
        404,
    )


@bp.route("/download_gif/<session_id>", methods=["GET"])
def download_gif(session_id):
    """API to download the GIF file of Overlay."""
    gif_filename = "serie_0.gif"
    gif_path = get_file_path(session_id, gif_filename)

    if os.path.exists(gif_path):
        print(f"✅ GIF found: {gif_path}, preparing download...")
        return send_file(gif_path, as_attachment=True)

    print(f"⚠️ GIF not found: {gif_path}")
    return (
        jsonify(
            {
                "error": "GIF file not found",
                "session_id": session_id,
                "filename": gif_filename,
            }
        ),
        404,
    )


@bp.route("/convert-list", methods=["POST"])
def convert_dicom_list():
    if "files" not in request.files:
        return jsonify({"error": "No files uploaded"}), 400

    files = request.files.getlist("files")

    if not files:
        return jsonify({"error": "Empty file list"}), 400

    result = []
    for file in files:
        try:
            img_base64 = dicom_to_png(file)
            result.append(
                {"filename": f"{file.filename}.png", "image_base64": img_base64}
            )
        except Exception as e:
            return (
                jsonify({"error": f"Error processing file {file.filename}: {str(e)}"}),
                500,
            )

    return jsonify({"images": result})
