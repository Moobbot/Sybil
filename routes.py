import os
import shutil
import uuid
import zipfile

from flask import Blueprint, jsonify, request, send_file, send_from_directory

from call_model import load_model, predict
from config import PREDICTION_CONFIG, PYTHON_ENV, RESULTS_FOLDER, UPLOAD_FOLDER
from utils import (
    create_zip_result,
    dicom_to_png,
    extract_zip_file,
    get_file_path,
    get_overlay_files,
    get_valid_files,
    save_uploaded_files,
    save_uploaded_zip,
)

bp = Blueprint("routes", __name__)

model = load_model()


@bp.route("/api_predict", methods=["POST"])
def api_predict():
    """
    API nhận session_id, truy cập folder đã giải nén sẵn, chạy model và trả về kết quả
    Args:
        session_id (str): ID phiên làm việc, trỏ tới folder đã giải nén sẵn
    Returns:
        JSON: Kết quả dự đoán gồm link tải ZIP và attention info
    """
    data = request.get_json()
    session_id = data.get("session_id") if data else None
    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400

    unzip_path = os.path.join(UPLOAD_FOLDER, session_id)
    if not os.path.exists(unzip_path):
        return jsonify({"error": f"Session folder not found: {unzip_path}"}), 404

    valid_files = get_valid_files(unzip_path)
    if not valid_files:
        return jsonify({"error": "No valid files found in the session folder"}), 400

    output_dir = os.path.join(RESULTS_FOLDER, session_id, "sybil")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Session ID: {session_id}, Output directory: {output_dir}")

    # Run prediction
    pred_dict, _, attention_info = predict(unzip_path, output_dir, model)

    # KHÔNG xoá thư mục sau khi tạo zip (giữ lại cho các request khác)
    response = {
        "session_id": session_id,
        "predictions": pred_dict["predictions"],
        "attention_info": attention_info,
        "message": "Prediction successful.",
    }
    if PYTHON_ENV == "develop":
        print(f"Response: {response}")
    return jsonify(response)


@bp.route("/api_predict_v1", methods=["POST"])
def api_predict_v1():
    """API nhận session_id, truy cập folder đã giải nén sẵn, chạy model và trả về kết quả"""
    data = request.get_json()
    session_id = data.get("session_id") if data else None
    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400

    upload_path = os.path.join(UPLOAD_FOLDER, session_id)
    if not os.path.exists(upload_path):
        return jsonify({"error": f"Session folder not found: {upload_path}"}), 404

    valid_files = get_valid_files(upload_path)
    if not valid_files:
        return jsonify({"error": "No valid files found in the session folder"}), 400

    output_dir = os.path.join(RESULTS_FOLDER, session_id, "sybil")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Session ID: {session_id}, Output directory: {output_dir}")

    # Run prediction
    pred_dict, overlayed_images, attention_info = predict(
        upload_path,
        output_dir,
        model,
        visualize_attentions_img=True,
        save_as_dicom=True,
    )

    response = {
        "session_id": session_id,
        "predictions": pred_dict["predictions"],
        "attention_info": attention_info,
        "message": "Prediction successful.",
    }
    return jsonify(response)


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
