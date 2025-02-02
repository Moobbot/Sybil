import pickle
import typing
import uuid
import os
import json
import time
import shutil
import zipfile
from flask import Flask, request, jsonify, send_file, send_from_directory
import urllib
from werkzeug.utils import secure_filename
from typing import Literal
from sybil.utils import logging_utils
from sybil.datasets import utils
from sybil.serie import Serie
from sybil.model import Sybil
from sybil.utils.visualization import visualize_attentions

# CODE ĐỂ BIẾT CHƯƠNG TRINH CHAY TỪ a-b, CÁC THAM SỐ GỐC

# Cấu hình ứng dụng Flask
app = Flask(__name__)

# Cấu hình thư mục tải lên và lưu trữ kết quả
UPLOAD_FOLDER = "uploads"
RESULTS_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {"dcm", "png", "jpg", "jpeg"}

app.config.update(UPLOAD_FOLDER=UPLOAD_FOLDER, RESULTS_FOLDER=RESULTS_FOLDER)

from sybil import Sybil


CHECKPOINT_URL = "https://github.com/reginabarzilaygroup/Sybil/releases/download/v1.5.0/sybil_checkpoints.zip"
CHECKPOINT_DIR = "sybil_checkpoints"

# Đường dẫn đến các checkpoint và calibrator
model_paths = [
    os.path.join(CHECKPOINT_DIR, "28a7cd44f5bcd3e6cc760b65c7e0d54d.ckpt"),
    os.path.join(CHECKPOINT_DIR, "56ce1a7d241dc342982f5466c4a9d7ef.ckpt"),
    os.path.join(CHECKPOINT_DIR, "64a91b25f84141d32852e75a3aec7305.ckpt"),
    os.path.join(CHECKPOINT_DIR, "65fd1f04cb4c5847d86a9ed8ba31ac1a.ckpt"),
    os.path.join(CHECKPOINT_DIR, "624407ef8e3a2a009f9fa51f9846fe9a.ckpt"),
]

calibrator_path = os.path.join(CHECKPOINT_DIR, "sybil_ensemble_simple_calibrator.json")


def download_checkpoints():
    """Download and extract checkpoint if not exist."""
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

        print(f"Downloading checkpoints from {CHECKPOINT_URL}...")
        zip_path = os.path.join(CHECKPOINT_DIR, "sybil_checkpoints.zip")

        urllib.request.urlretrieve(CHECKPOINT_URL, zip_path)

        # Giải nén file
        print("Extracting checkpoints...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(CHECKPOINT_DIR)

        os.remove(zip_path)  # Xóa file zip sau khi giải nén
        print("Checkpoints downloaded and extracted successfully.")


def load_model(name_or_path=None, calibrator=None):
    """
    Load a trained Sybil model from a checkpoint or a directory of checkpoints.
    If there is no model or calibrator, download from CHECKPOINT_URL.

    Args:
        name_or_path (str | list): The path to the checkpoint file or the checkpoint file list.
        calibrator (str): The path to the Calibrator file.

    Returns:
        Sybil: Model object has loaded.
    """
    # Kiểm tra và tải checkpoints nếu cần
    if name_or_path is None or not all(os.path.exists(p) for p in name_or_path):
        print("Model checkpoints not found. Downloading...")
        download_checkpoints()
        name_or_path = model_paths  # Gán lại đường dẫn sau khi tải xuống

    if calibrator is None or not os.path.exists(calibrator):
        print("Calibrator file not found. Downloading...")
        download_checkpoints()
        calibrator = calibrator_path  # Gán lại đường dẫn sau khi tải xuống

    # Load model từ checkpoint
    print("Loading Sybil model...")
    model = Sybil(name_or_path=name_or_path, calibrator_path=calibrator)
    print("Model loaded successfully.")

    return model


def predict(
    image_dir,
    output_dir,
    model_name="sybil_ensemble",
    return_attentions=False,
    write_attention_images=False,
    file_type: Literal["auto", "dicom", "png"] = "auto",
    threads: int = 0,
):
    logger = logging_utils.get_logger()

    return_attentions |= write_attention_images

    input_files = os.listdir(image_dir)
    input_files = [
        os.path.join(image_dir, x) for x in input_files if not x.startswith(".")
    ]
    input_files = [x for x in input_files if os.path.isfile(x)]

    voxel_spacing = None
    if file_type == "auto":
        extensions = {os.path.splitext(x)[1] for x in input_files}

        if not extensions:
            raise ValueError("No files with valid extensions found in the directory.")

        extension = extensions.pop()
        if len(extensions) > 1:
            raise ValueError(
                f"Multiple file types found in {image_dir}: {','.join(extensions)}"
            )

        file_type = "dicom"
        if extension.lower() in {".png", "png"}:
            file_type = "png"
            voxel_spacing = utils.VOXEL_SPACING
            logger.debug(f"Using default voxel spacing: {voxel_spacing}")
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
    # Get risk scores
    serie = Serie(input_files, voxel_spacing=voxel_spacing, file_type=file_type)
    series = [serie]
    prediction = model.predict(
        series, return_attentions=return_attentions, threads=threads
    )
    prediction_scores = prediction.scores[0]

    logger.debug(f"Prediction finished. Results:\n{prediction_scores}")

    prediction_path = os.path.join(output_dir, "prediction_scores.json")
    pred_dict = {"predictions": prediction.scores}
    with open(prediction_path, "w") as f:
        json.dump(pred_dict, f, indent=2)

    series_with_attention = None
    if return_attentions:
        attention_path = os.path.join(output_dir, "attention_scores.pkl")
        with open(attention_path, "wb") as f:
            pickle.dump(prediction, f)

    if write_attention_images:
        series_with_attention = visualize_attentions(
            series,
            attentions=prediction.attentions,
            save_directory=output_dir,
            gain=3,
        )

    return pred_dict, series_with_attention


def allowed_file(filename):
    """Kiểm tra tệp có định dạng phù hợp không"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def cleanup_old_results(expiry_time=3600):
    """Xóa thư mục cũ sau khoảng thời gian nhất định (3600 giây = 1 giờ)"""
    current_time = time.time()
    for folder in os.listdir(app.config["RESULTS_FOLDER"]):
        folder_path = os.path.join(app.config["RESULTS_FOLDER"], folder)
        if (
            os.path.isdir(folder_path)
            and (current_time - os.path.getmtime(folder_path)) > expiry_time
        ):
            shutil.rmtree(folder_path)
            print(f"Deleted old result folder: {folder_path}")


# Chạy dọn dẹp mỗi khi ứng dụng khởi động
cleanup_old_results()

model = load_model(model_paths, calibrator_path)


@app.route("/api_predict", methods=["POST"])
def api_predict():
    """API nhận ảnh, chạy mô hình và trả về dự đoán và overlayed_images"""

    print("API predict called")

    files = request.files.getlist("file")  # Nhận tất cả các tệp tải lên
    if not files or all(file.filename == "" for file in files):
        return jsonify({"error": "No selected files"}), 400

    # Tạo UUID cho mỗi yêu cầu dự đoán
    session_id = str(uuid.uuid4())

    uploaded_files = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            folder_file_path = os.path.join(app.config["UPLOAD_FOLDER"], session_id)
            os.makedirs(folder_file_path, exist_ok=True)

            file_path = os.path.join(folder_file_path, filename)
            file.save(file_path)
            uploaded_files.append(file_path)
            print("Uploaded file:", filename)
        else:
            return (
                jsonify({"error": "Invalid file type. Only DICOM or PNG allowed."}),
                400,
            )

    if not uploaded_files:
        return jsonify({"error": "No valid files uploaded"}), 400

    output_dir = os.path.join(app.config["RESULTS_FOLDER"], session_id)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Session ID: {session_id}, Output directory: {output_dir}")

    # Chạy dự đoán với tất cả các file đã upload
    pred_dict, overlayed_images = predict(
        folder_file_path, output_dir, write_attention_images=True
    )

    # Truy cập thư mục serie_0 để lấy ảnh overlay
    overlay_dir = os.path.join(output_dir, "serie_0")
    overlay_files = [f for f in os.listdir(overlay_dir) if f.endswith(".png")]

    if not overlay_files:
        print("No overlay images found.")

    # Tạo danh sách các URL tải xuống và xem trước ảnh
    base_url = request.host_url.rstrip("/")
    overlay_image_info = []

    for img in overlay_files:
        overlay_image_info.append(
            {
                "filename": img,
                "download_link": f"{base_url}/download/{session_id}/{img}",
                "preview_link": f"{base_url}/preview/{session_id}/{img}",
            }
        )

    # Trả về kết quả JSON bao gồm đường dẫn và attention values
    response = {
        "session_id": session_id,
        "predictions": pred_dict["predictions"],
        "overlay_images": overlay_image_info,
        "gif_download": f"{base_url}/download_gif/{session_id}",
        "message": "Prediction successful. Download overlay images using the provided links.",
    }

    return jsonify(response)


@app.route("/download/<session_id>/<filename>", methods=["GET"])
def download_file(session_id, filename):
    """API để tải xuống ảnh overlay theo session ID"""
    file_path = os.path.join(
        app.config["RESULTS_FOLDER"], session_id, "serie_0", filename
    )
    return (
        send_file(file_path, as_attachment=True)
        if os.path.exists(file_path)
        else jsonify({"error": "File not found"})
    ), 404


@app.route("/preview/<session_id>/<filename>", methods=["GET"])
def preview_file(session_id, filename):
    """API để xem trước ảnh overlay trực tiếp trên trình duyệt"""
    overlay_dir = os.path.join(app.config["RESULTS_FOLDER"], session_id, "serie_0")
    return (
        send_from_directory(overlay_dir, filename)
        if os.path.exists(os.path.join(overlay_dir, filename))
        else jsonify({"error": "File not found"})
    ), 404


@app.route("/download_gif/<session_id>", methods=["GET"])
def download_gif(session_id):
    """API để tải xuống file GIF của ảnh overlay"""
    gif_filename = "serie_0.gif"
    gif_path = os.path.join(
        app.config["RESULTS_FOLDER"], session_id, "serie_0", gif_filename
    )

    print(f"Checking GIF path: {gif_path}")  # Debugging

    if os.path.exists(gif_path):
        return send_file(gif_path, as_attachment=True)

    return jsonify({"error": "GIF file not found"}), 404


if __name__ == "__main__":
    app.run(debug=True)
