import base64
import io
import os
import time
import shutil
import socket
import zipfile
from flask import jsonify
import numpy as np
import pydicom
from PIL import Image
from werkzeug.utils import secure_filename
from config import PYTHON_ENV, RESULTS_FOLDER, UPLOAD_FOLDER, FILE_RETENTION


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {
        "dcm",
        "png",
        "jpg",
        "jpeg",
    }


def cleanup_old_results(folders, expiry_time=FILE_RETENTION):
    """
    Delete the old folder after a certain period of time.

    Args:
        folders (List): List of folders to be checked.
        expiry_time (int): expiry period (second). The default is 3600 seconds (1 hour).
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


def save_uploaded_files(files, session_id, folder_save=UPLOAD_FOLDER):
    """Save uploaded files to the specified folder

    Args:
        files (list): List of uploaded files
        session_id (str): Session ID for the upload
        folder_save (str): Folder to save the files to

    Returns:
        tuple: (list of saved files, path to saved files)
    """
    uploaded_files = []
    upload_path = os.path.join(folder_save, session_id)
    os.makedirs(upload_path, exist_ok=True)

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(upload_path, filename)
            file.save(file_path)
            uploaded_files.append(filename)

    return uploaded_files, upload_path


def get_file_path(session_id, filename):
    """Trả về đường dẫn đầy đủ của file trong thư mục kết quả."""
    return os.path.join(RESULTS_FOLDER, session_id, "serie_0", filename)


def get_overlay_files(output_dir, session_id):
    """Lấy danh sách ảnh overlay trong thư mục session."""
    if not os.path.exists(output_dir) or not os.listdir(output_dir):
        print(f"⚠️ No overlay images found for session {session_id}")
        return []

    return [
        img
        for img in os.listdir(output_dir)
        if os.path.isfile(os.path.join(output_dir, img)) and img.endswith('.dcm')
    ]


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 1))
        ip_address = s.getsockname()[0]
        s.close()
    except:
        ip_address = "127.0.0.1"
    return ip_address


def save_uploaded_zip(file, session_id, folder_save=UPLOAD_FOLDER):
    """Lưu file ZIP tải lên"""
    zip_path = os.path.join(folder_save, f"{session_id}.zip")
    file.save(zip_path)
    return zip_path


def extract_zip_file(zip_path, session_id, folder_save=UPLOAD_FOLDER):
    """Giải nén ZIP, kiểm tra thư mục con"""
    unzip_path = os.path.join(folder_save, session_id)
    os.makedirs(unzip_path, exist_ok=True)
    print("unzip_path:", unzip_path)

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(unzip_path)
    except zipfile.BadZipFile:
        os.remove(zip_path)
        return None, jsonify({"error": "Invalid ZIP file"}), 400

    os.remove(zip_path)

    # Nếu ZIP chỉ có 1 thư mục con, cập nhật lại đường dẫn
    subfolders = [
        f for f in os.listdir(unzip_path) if os.path.isdir(os.path.join(unzip_path, f))
    ]
    if len(subfolders) == 1:
        unzip_path = os.path.join(unzip_path, subfolders[0])

    return unzip_path, None, None


def get_valid_files(unzip_path):
    """Lấy danh sách file hợp lệ (DICOM/PNG)"""
    valid_files = []
    for root, _, files in os.walk(unzip_path):
        for filename in files:
            if filename.lower().endswith((".dcm", ".png")):
                valid_files.append(os.path.join(root, filename))
    return valid_files


def create_zip_result(output_dir, session_id, folder_save=RESULTS_FOLDER):
    """Nén ảnh dự đoán thành file ZIP"""
    result_zip_path = os.path.join(folder_save, f"{session_id}.zip")
    if PYTHON_ENV == "develop":
        print(f"Creating zip file from {output_dir} to {result_zip_path}")

    with zipfile.ZipFile(result_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, output_dir)
                zipf.write(file_path, arcname)

    if PYTHON_ENV == "develop":
        print(f"Zip file size: {os.path.getsize(result_zip_path)} bytes")
    return result_zip_path
