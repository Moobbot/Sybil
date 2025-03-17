import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pydicom


# Kiểm tra loại mã hóa
ENCODING_DICT = {
    "1.2.840.10008.1.2": "Little Endian Implicit (Không nén)",
    "1.2.840.10008.1.2.1": "Little Endian Explicit (Không nén)",
    "1.2.840.10008.1.2.2": "Big Endian Explicit (Không nén)",
    "1.2.840.10008.1.2.4.50": "JPEG Baseline (Lossy)",
    "1.2.840.10008.1.2.4.51": "JPEG Extended",
    "1.2.840.10008.1.2.4.57": "JPEG Lossless (Extended: Process 14)",
    "1.2.840.10008.1.2.4.70": "JPEG Lossless (SV1)",
    "1.2.840.10008.1.2.4.90": "JPEG 2000 Lossless",
    "1.2.840.10008.1.2.4.91": "JPEG 2000 (Lossy & Lossless)",
    "1.2.840.10008.1.2.5": "RLE Lossless",
    "1.2.840.10008.1.2.4.100": "MPEG-2 Main Profile",
    "1.2.840.10008.1.2.4.102": "MPEG-4 AVC/H.264",
    "1.2.840.10008.1.2.4.103": "MPEG-4 BD-Compatible",
    "1.2.840.10008.1.2.4.106": "HEVC/H.265",
}


def check_dicom_color(dicom_path):
    try:
        # Đọc file DICOM
        dicom_file = pydicom.dcmread(dicom_path)

        # Lấy thông tin về kiểu ảnh
        photometric_interpretation = dicom_file.PhotometricInterpretation
        samples_per_pixel = dicom_file.SamplesPerPixel

        print(f"📂 File: {dicom_path}")
        print(f"🖼️ Photometric Interpretation: {photometric_interpretation}")
        print(f"🎨 Samples Per Pixel: {samples_per_pixel}")

        # Xác định loại ảnh
        if photometric_interpretation in ["RGB", "YBR_FULL", "YBR_PARTIAL_422"]:
            print("✅ Ảnh DICOM là ảnh MÀU 🎨")
        elif photometric_interpretation in ["MONOCHROME1", "MONOCHROME2"]:
            print("✅ Ảnh DICOM là ảnh GRAYSCALE ⚫⚪")
        else:
            print("❓ Không xác định được loại ảnh")

        return photometric_interpretation, samples_per_pixel

    except Exception as e:
        print(f"❌ Lỗi khi đọc file DICOM: {str(e)}")
        return None, None


def check_dicom_encoding(dicom_path):
    try:
        # Đọc file DICOM
        dicom_file = pydicom.dcmread(dicom_path, stop_before_pixels=True)

        # Lấy Transfer Syntax UID (kiểu mã hóa)
        transfer_syntax_uid = dicom_file.file_meta.TransferSyntaxUID
        print(f"📂 File: {dicom_path}")
        print(f"🔍 Transfer Syntax UID: {transfer_syntax_uid}")

        # Kiểm tra và hiển thị kiểu mã hóa
        encoding_type = ENCODING_DICT.get(str(transfer_syntax_uid), "🔴 Không xác định")
        print(f"✅ Kiểu mã hóa: {encoding_type}")

        return str(transfer_syntax_uid), encoding_type

    except Exception as e:
        print(f"❌ Lỗi khi đọc file DICOM: {str(e)}")
        return None, None


def read_dicom_metadata(dicom_path):
    try:
        # Đọc file DICOM
        dicom_file = pydicom.dcmread(dicom_path)

        # Lấy các thông số quan trọng
        metadata = {
            "File Name": dicom_path,
            "Patient Name": dicom_file.get("PatientName", "N/A"),
            "Patient ID": dicom_file.get("PatientID", "N/A"),
            "Patient Age": dicom_file.get("PatientAge", "N/A"),
            "Patient Sex": dicom_file.get("PatientSex", "N/A"),
            "Study Date": dicom_file.get("StudyDate", "N/A"),
            "Modality": dicom_file.get("Modality", "N/A"),
            "Institution Name": dicom_file.get("InstitutionName", "N/A"),
            "Manufacturer": dicom_file.get("Manufacturer", "N/A"),
            "Model": dicom_file.get("ManufacturerModelName", "N/A"),
            "Slice Thickness": dicom_file.get("SliceThickness", "N/A"),
            "Pixel Spacing": dicom_file.get("PixelSpacing", "N/A"),
            "Image Dimensions": f"{dicom_file.Rows} x {dicom_file.Columns}",
            "Bits Stored": dicom_file.get("BitsStored", "N/A"),
            "Photometric Interpretation": dicom_file.get(
                "PhotometricInterpretation", "N/A"
            ),
            "Samples Per Pixel": dicom_file.get("SamplesPerPixel", "N/A"),
            "Transfer Syntax UID": dicom_file.file_meta.TransferSyntaxUID,
        }

        # In ra thông tin
        # print("\n📂 DICOM Metadata:")
        # for key, value in metadata.items():
        #     print(f"🔹 {key}: {value}")

        return metadata

    except Exception as e:
        print(f"❌ Lỗi khi đọc file DICOM: {str(e)}")
        return None

# Hàm hiển thị ảnh DICOM
def show_dicom_image(dicom_path):
    try:
        dicom_file = pydicom.dcmread(dicom_path)
        photometric_interpretation = dicom_file.PhotometricInterpretation
        image = dicom_file.pixel_array

        # Nếu ảnh màu YBR_FULL, cần chuyển sang grayscale
        if photometric_interpretation == "YBR_FULL":
            image = np.dot(image, [0.299, 0.587, 0.114])

        # Hiển thị ảnh
        plt.imshow(image, cmap="gray" if len(image.shape) == 2 else None)
        plt.axis("off")
        plt.title(os.path.basename(dicom_path))
        plt.show()
    except Exception as e:
        print(f"❌ Không thể hiển thị ảnh DICOM: {str(e)}")


def view_dicom_info(dicom_paths):
    """
    Kiểm tra và hiển thị thông tin của danh sách file DICOM
    """
    for dicom_path in dicom_paths:
        print("\n" + "-" * 50)
        check_dicom_encoding(dicom_path)
        print("-" * 50)
        check_dicom_color(dicom_path)
        print("-" * 50)
        read_dicom_metadata(dicom_path)
        print("-" * 50)
        show_dicom_image(dicom_path)
        print("=" * 50)


# Hàm đọc toàn bộ DICOM trong thư mục và lưu vào Excel
def process_dicom_folder(folder_path, output_excel="metadata.xlsx"):
    dicom_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".dcm")
    ]

    if not dicom_files:
        print("❌ Không tìm thấy file DICOM nào trong thư mục.")
        return

    print(f"🔍 Đang xử lý {len(dicom_files)} file DICOM...")

    # Đọc metadata của tất cả file
    metadata_list = [read_dicom_metadata(dicom_file) for dicom_file in dicom_files]

    # Chuyển dữ liệu thành DataFrame
    df = pd.DataFrame(metadata_list)

    # Xuất ra file Excel
    df.to_excel(output_excel, index=False)
    print(f"✅ Dữ liệu đã được lưu vào {output_excel}")

    return df


# Chạy xử lý trên một thư mục DICOM
dicom_folder = r"D:\Work\Clients\Job\dicom-sybil\Sybil\custom\test\580adef9-719c-4935-9dc9-a9ed7f8aba04"
metadata_df = process_dicom_folder(dicom_folder)

# Ví dụ sử dụng
dicom_files = [
    r"D:\Work\Clients\Job\dicom-sybil\Sybil\custom\test\a2dcc9ee-a590-4885-9567-e33b092f87de\slice_148.dcm",
    r"D:\Work\Clients\Job\dicom-sybil\Sybil\custom\test\oke_view\slice_146.dcm"
]

# Gọi hàm tổng hợp
# view_dicom_info(dicom_files)