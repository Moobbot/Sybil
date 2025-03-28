import base64
import io
import json
import os
import pickle
import shutil
import socket
import time
import typing
import uuid
import urllib
import zipfile

from typing import Literal

import numpy as np
import pydicom
from PIL import Image
from flask import Flask, jsonify, request, send_file, send_from_directory
from werkzeug.utils import secure_filename

from sybil.datasets import utils
from sybil.model import Sybil
from sybil.serie import Serie
from sybil.utils import logging_utils
from sybil.utils.visualization import visualize_attentions

# CODE ĐỂ BIẾT CHƯƠNG TRINH CHAY TỪ a-b, CÁC THAM SỐ GỐC

# Cấu hình ứng dụng Flask
app = Flask(__name__)

# Cấu hình thư mục tải lên và lưu trữ kết quả
UPLOAD_FOLDER = "uploads"
RESULTS_FOLDER = "results"
CHECKPOINT_DIR = "sybil_checkpoints"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"dcm", "png", "jpg", "jpeg"}
app.config.update(UPLOAD_FOLDER=UPLOAD_FOLDER, RESULTS_FOLDER=RESULTS_FOLDER)

CHECKPOINT_URL = "https://github.com/reginabarzilaygroup/Sybil/releases/download/v1.5.0/sybil_checkpoints.zip"

# Đường dẫn đến các checkpoint và calibrator
MODEL_PATHS = [
    os.path.join(CHECKPOINT_DIR, f"{model}.ckpt")
    for model in [
        "28a7cd44f5bcd3e6cc760b65c7e0d54d",
        "56ce1a7d241dc342982f5466c4a9d7ef",
        "64a91b25f84141d32852e75a3aec7305",
        "65fd1f04cb4c5847d86a9ed8ba31ac1a",
        "624407ef8e3a2a009f9fa51f9846fe9a",
    ]
]
calibrator_path = os.path.join(CHECKPOINT_DIR, "sybil_ensemble_simple_calibrator.json")


def allowed_file(filename):
    """Kiểm tra tệp có định dạng hợp lệ."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def download_checkpoints():
    """Download and extract checkpoint if not exist."""
    if not os.path.exists(CHECKPOINT_DIR) or not all(
        os.path.exists(p) for p in MODEL_PATHS
    ):
        print(f"Downloading checkpoints from {CHECKPOINT_URL}...")
        zip_path = os.path.join(CHECKPOINT_DIR, "sybil_checkpoints.zip")
        urllib.request.urlretrieve(CHECKPOINT_URL, zip_path)

        print("Extracting checkpoints...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(CHECKPOINT_DIR)
        os.remove(zip_path)
        print("Checkpoints downloaded and extracted successfully.")


def cleanup_old_results(folders, expiry_time=3600):
    """
    Xóa thư mục cũ sau một khoảng thời gian nhất định.

    Args:
        folders (list): Danh sách các thư mục cần kiểm tra.
        expiry_time (int): Thời gian hết hạn (giây). Mặc định là 3600 giây (1 giờ).
    """
    current_time = time.time()
    for folder in folders:
        if os.path.exists(folder):
            for subfolder in os.listdir(folder):
                subfolder_path = os.path.join(folder, subfolder)
                if (
                    os.path.isdir(subfolder_path)
                    and (current_time - os.path.getmtime(subfolder_path)) > expiry_time
                ):
                    shutil.rmtree(subfolder_path)
                    print(f"Deleted old folder: {subfolder_path}")


def save_uploaded_files(files, session_id):
    """Lưu các file tải lên theo session_id."""
    upload_path = os.path.join(UPLOAD_FOLDER, session_id)
    os.makedirs(upload_path, exist_ok=True)
    uploaded_files = []

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(upload_path, filename)
            file.save(file_path)
            uploaded_files.append(file_path)
            print(f"Uploaded file: {filename}")

    return uploaded_files, upload_path


def load_model(name_or_paths="sybil_ensemble", calibrator_path=None):
    """
    Load a trained Sybil model from a checkpoint or a directory of checkpoints.
    If there is no model or calibrator, download from CHECKPOINT_URL.

    Args:
        name_or_paths (str | list): The path to the checkpoint file or the checkpoint file list.
        calibrator (str): The path to the Calibrator file.

    Returns:
        Sybil: Model object has loaded.
    """
    # Kiểm tra và tải checkpoints nếu cần
    if (
        name_or_paths is None
        or not all(os.path.exists(p) for p in name_or_paths)
        or calibrator_path is None
        or not os.path.exists(calibrator_path)
    ):
        print("Model and Calibrator checkpoints not found. Downloading...")
        download_checkpoints()
        name_or_paths = MODEL_PATHS  # Gán lại đường dẫn sau khi tải xuống

    # Load model từ checkpoint
    print("Loading Sybil model...")
    model = Sybil(name_or_path=name_or_paths, calibrator_path=calibrator_path)
    print("Model loaded successfully.")

    return model


# Dọn dẹp thư mục khi khởi động ứng dụng
cleanup_old_results([UPLOAD_FOLDER, RESULTS_FOLDER])

# Tải mô hình khi khởi động server
model = load_model(MODEL_PATHS, calibrator_path)


def predict(
    image_dir,
    output_dir,
    model_name="sybil_ensemble",
    return_attentions=True,
    visualize_attentions_img=False,
    save_as_dicom=False,  # Thêm lựa chọn lưu ảnh dưới dạng DICOM
    file_type: Literal["auto", "dicom", "png"] = "auto",
    threads: int = 0,
):
    """Chạy mô hình dự đoán."""
    logger = logging_utils.get_logger()

    input_files = [
        os.path.join(image_dir, x)
        for x in os.listdir(image_dir)
        if os.path.isfile(os.path.join(image_dir, x))
    ]

    if not input_files:
        raise ValueError("⚠️ No valid files found in the directory.")

    voxel_spacing = None
    if file_type == "auto":
        extensions = {os.path.splitext(x)[1] for x in input_files}
        if not extensions:
            raise ValueError("⚠️ No files with valid extensions found.")
        extension = extensions.pop()
        if len(extensions) > 1:
            raise ValueError(
                f"⚠️ Multiple file types found in {image_dir}: {','.join(extensions)}"
            )

        file_type = "dicom" if extension.lower() not in {".png"} else "png"
        if file_type == "png":
            voxel_spacing = utils.VOXEL_SPACING
            logger.debug(f"Using default voxel spacing: {voxel_spacing}")

    logger.debug(f"Processing {len(input_files)} {file_type} files from {image_dir}")

    assert file_type in {"dicom", "png"}
    file_type = typing.cast(typing.Literal["dicom", "png"], file_type)

    num_files = len(input_files)

    logger.debug(
        f"Beginning prediction using {num_files} {file_type} files from {image_dir}"
    )

    # print("Load model")
    # Load a trained model
    # model = Sybil(model_name)
    # print("model loaded")
    # Tạo đối tượng Serie - Get risk scores
    serie = Serie(input_files, voxel_spacing=voxel_spacing, file_type=file_type)
    prediction = model.predict(
        [serie], return_attentions=return_attentions, threads=threads
    )
    prediction_scores = prediction.scores[0]

    logger.debug(f"Prediction finished. Results:\n{prediction_scores}")

    # Lưu kết quả dự đoán
    prediction_path = os.path.join(output_dir, "prediction_scores.json")
    pred_dict = {"predictions": prediction.scores}
    with open(prediction_path, "w") as f:
        json.dump(pred_dict, f, indent=2)

    series_with_attention = None
    if return_attentions:
        attention_path = os.path.join(output_dir, "attention_scores.pkl")
        with open(attention_path, "wb") as f:
            pickle.dump(prediction, f)

    # Nếu chọn lưu ảnh DICOM, lấy metadata của từng ảnh
    dicom_metadata_list = []
    if save_as_dicom and file_type == "dicom":
        dicom_metadata_list = [pydicom.dcmread(f) for f in input_files]

    # Gọi visualize_attentions với danh sách metadata riêng
    if visualize_attentions_img:
        series_with_attention = visualize_attentions(
            [serie],
            attentions=prediction.attentions,
            save_directory=output_dir,
            gain=3,
            save_as_dicom=save_as_dicom,
            dicom_metadata_list=dicom_metadata_list,  # Update metadata
        )

    return pred_dict, series_with_attention


def get_overlay_files(output_dir, session_id):
    """Lấy danh sách ảnh overlay trong thư mục 'serie_0'."""
    overlay_dir = os.path.join(output_dir, "serie_0")

    if not os.path.exists(overlay_dir) or not os.listdir(overlay_dir):
        print(f"⚠️ No overlay images found for session {session_id}")
        return []

    return [
        img
        for img in os.listdir(overlay_dir)
        if os.path.isfile(os.path.join(overlay_dir, img))
    ]


def dicom_to_png(dicom_file):
    """Chuyển đổi file DICOM thành ảnh PNG và trả về ảnh dạng base64"""
    dicom_data = pydicom.dcmread(dicom_file)

    # Kiểm tra loại ảnh
    photometric_interpretation = dicom_data.PhotometricInterpretation
    pixel_array = dicom_data.pixel_array.astype(np.float32)

    # Chuẩn hóa giá trị pixel về khoảng 0-255
    pixel_array = (
        (pixel_array - np.min(pixel_array))
        / (np.max(pixel_array) - np.min(pixel_array))
        * 255
    )
    pixel_array = pixel_array.astype(np.uint8)

    # Xử lý ảnh màu
    if photometric_interpretation == "RGB":
        image = Image.fromarray(pixel_array)
    elif photometric_interpretation == "YBR_FULL":
        image = Image.fromarray(pixel_array, mode="YCbCr").convert("RGB")
    else:
        image = Image.fromarray(pixel_array, mode="L")  # Ảnh grayscale

    # Lưu ảnh vào bộ nhớ dưới dạng PNG
    img_io = io.BytesIO()
    image.save(img_io, format="PNG")
    img_io.seek(0)

    # Mã hóa ảnh PNG thành base64
    img_base64 = base64.b64encode(img_io.getvalue()).decode("utf-8")

    return img_base64


@app.route("/api_predict", methods=["POST"])
def api_predict():
    """API để nhận ảnh, chạy mô hình, và trả về dự đoán."""

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
        upload_path, output_dir, visualize_attentions_img=True, save_as_dicom=True
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


def get_file_path(session_id, filename):
    """Trả về đường dẫn đầy đủ của file trong thư mục kết quả."""
    return os.path.join(RESULTS_FOLDER, session_id, "serie_0", filename)


@app.route("/download/<session_id>/<filename>", methods=["GET"])
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


@app.route("/preview/<session_id>/<filename>", methods=["GET"])
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


@app.route("/download_gif/<session_id>", methods=["GET"])
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


@app.route("/convert-list", methods=["POST"])
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


def get_local_ip():
    """Get the local IP address of the machine."""
    try:
        # Connect to an external host to get the local IP address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0)
        s.connect(("8.8.8.8", 1))  # Example using Google DNS
        IP_CONNECT = s.getsockname()[0]
        s.close()
    except Exception:
        IP_CONNECT = "127.0.0.1"
    return IP_CONNECT


if __name__ == "__main__":
    LOCAL_IP = get_local_ip()
    HOST_CONNECT = (
        "0.0.0.0"  # Chạy trên tất cả các IP, bao gồm cả localhost và IP cục bộ
    )
    PORT_CONNECT = 5000
    print(f"Running on: http://127.0.0.1:{PORT_CONNECT} (localhost)")
    print(f"Running on: http://{LOCAL_IP}:{PORT_CONNECT} (local network)")

    # Chạy trên tất cả địa chỉ IP (0.0.0.0) để nhận cả localhost và IP cục bộ
    app.run(host=HOST_CONNECT, port=PORT_CONNECT, debug=True)
