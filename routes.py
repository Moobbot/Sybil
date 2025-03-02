import os
import uuid
import zipfile
import shutil
from flask import Blueprint, request, jsonify, send_file, send_from_directory
from config import RESULTS_FOLDER, UPLOAD_FOLDER
from call_model import load_model, predict
from utils import dicom_to_png, get_overlay_files, save_uploaded_files, get_file_path

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
    print(file)
    # Tạo UUID cho mỗi yêu cầu dự đoán
    session_id = str(uuid.uuid4())

    # Lưu file ZIP tải lên
    zip_path = os.path.join(UPLOAD_FOLDER, f"{session_id}.zip")
    file.save(zip_path)

    # Thư mục để giải nén file ZIP
    unzip_path = os.path.join(UPLOAD_FOLDER, session_id)
    print("unzip_path:", unzip_path)
    os.makedirs(unzip_path, exist_ok=True)

    # Thư mục để giải nén file ZIP
    unzip_path = os.path.join(UPLOAD_FOLDER, session_id)
    os.makedirs(unzip_path, exist_ok=True)

    # Giải nén file ZIP
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(unzip_path)
    except zipfile.BadZipFile:
        return jsonify({"error": "Invalid ZIP file"}), 400

    # Xóa file ZIP sau khi giải nén
    os.remove(zip_path)

    # Kiểm tra nếu ZIP giải nén tạo ra một thư mục con duy nhất
    subfolders = [f for f in os.listdir(unzip_path) if os.path.isdir(os.path.join(unzip_path, f))]
    if len(subfolders) == 1:
        unzip_path = os.path.join(unzip_path, subfolders[0])  # Cập nhật lại đường dẫn ảnh

    # Kiểm tra xem có file hợp lệ không
    valid_files = []
    for root, _, files in os.walk(unzip_path):
        for filename in files:
            if filename.lower().endswith((".dcm", ".png")):
                valid_files.append(os.path.join(root, filename))

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

    # Tạo file ZIP chứa ảnh kết quả
    result_zip_path = os.path.join(RESULTS_FOLDER, f"{session_id}.zip")
    with zipfile.ZipFile(result_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, output_dir)
                zipf.write(file_path, arcname)

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
        jsonify(
            {"error": "File not found", "session_id": session_id}
        ),
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
