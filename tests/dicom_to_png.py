import pydicom
import numpy as np
from PIL import Image
import os


def dicom_to_png(dicom_path, output_path):
    # Đọc file DICOM
    dicom_data = pydicom.dcmread(dicom_path)

    # Chuyển pixel array thành NumPy array
    pixel_array = dicom_data.pixel_array.astype(np.float32)

    # Chuẩn hóa giá trị pixel về khoảng 0-255
    pixel_array = (
        (pixel_array - np.min(pixel_array))
        / (np.max(pixel_array) - np.min(pixel_array))
        * 255
    )
    pixel_array = pixel_array.astype(np.uint8)

    # Chuyển đổi sang ảnh PNG
    image = Image.fromarray(pixel_array)
    image.save(output_path)

    print(f"Chuyển đổi thành công: {output_path}")


def dicom_to_png_color(dicom_path, output_path):
    # Đọc file DICOM
    dicom_data = pydicom.dcmread(dicom_path)

    # Kiểm tra loại ảnh
    photometric_interpretation = dicom_data.PhotometricInterpretation
    pixel_array = dicom_data.pixel_array

    # Chuẩn hóa giá trị pixel về khoảng 0-255
    pixel_array = pixel_array.astype(np.float32)
    pixel_array = (pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array)) * 255
    pixel_array = pixel_array.astype(np.uint8)

    # Xử lý ảnh màu
    if photometric_interpretation == "RGB":
        image = Image.fromarray(pixel_array)
    elif photometric_interpretation == "YBR_FULL":
        # Chuyển đổi từ YBR_FULL sang RGB
        image = Image.fromarray(pixel_array, mode="YCbCr").convert("RGB")
    else:
        # Nếu không phải RGB hoặc YBR, chuyển thành grayscale
        image = Image.fromarray(pixel_array, mode="L")

    # Lưu ảnh PNG
    image.save(output_path)
    print(f"Chuyển đổi thành công: {output_path}")


if __name__ == "__main__":
    dicom_path = "./results/f6defca5-3cbc-4703-8ef3-3323e3a98b59/serie_0"
    png_path = "./results/f6defca5-3cbc-4703-8ef3-3323e3a98b59/png"
    # Ví dụ sử dụng
    dicom_file = "./results/f6defca5-3cbc-4703-8ef3-3323e3a98b59/serie_0/slice_38.dcm"
    png_file = "./results/f6defca5-3cbc-4703-8ef3-3323e3a98b59/png/slice_38.png"
    for dicom_file in os.listdir(dicom_path):
        dicom_file_path = os.path.join(dicom_path, dicom_file)
        png_file_path = os.path.join(png_path, os.path.splitext(dicom_file)[0] + ".png")
        if not dicom_file_path.endswith(".dcm"):
            continue
        dicom_to_png_color(dicom_file_path, png_file_path)
