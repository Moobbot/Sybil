import base64
import io
import os
import time
import shutil
import socket
import numpy as np
import pydicom
from PIL import Image
from werkzeug.utils import secure_filename
from config import RESULTS_FOLDER, UPLOAD_FOLDER


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {
        "dcm",
        "png",
        "jpg",
        "jpeg",
    }


def cleanup_old_results(folders, expiry_time=3600):
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


def save_uploaded_files(files, session_id):
    """Save the files uploaded by session_id"""
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


def get_file_path(session_id, filename):
    """Trả về đường dẫn đầy đủ của file trong thư mục kết quả."""
    return os.path.join(RESULTS_FOLDER, session_id, "serie_0", filename)


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


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 1))
        ip_address = s.getsockname()[0]
        s.close()
    except:
        ip_address = "127.0.0.1"
    return ip_address
