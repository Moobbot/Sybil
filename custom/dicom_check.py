import os
import numpy as np
import pydicom
import pandas as pd
from matplotlib import pyplot as plt
from typing import List, Dict, Tuple, Optional


# Dictionary for encoding types
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


def check_dicom_color(dicom_path: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Check if a DICOM file is color or grayscale.

    Args:
        dicom_path (str): Path to the DICOM file

    Returns:
        Tuple[Optional[str], Optional[int]]: Tuple containing photometric interpretation and samples per pixel
    """
    try:
        dicom_file = pydicom.dcmread(dicom_path)
        photometric_interpretation = dicom_file.PhotometricInterpretation
        samples_per_pixel = dicom_file.SamplesPerPixel

        print(f"📂 File: {dicom_path}")
        print(f"🖼️ Photometric Interpretation: {photometric_interpretation}")
        print(f"🎨 Samples Per Pixel: {samples_per_pixel}")

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


def check_dicom_encoding(dicom_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Check the encoding type of a DICOM file.

    Args:
        dicom_path (str): Path to the DICOM file

    Returns:
        Tuple[Optional[str], Optional[str]]: Tuple containing transfer syntax UID and encoding type
    """
    try:
        dicom_file = pydicom.dcmread(dicom_path, stop_before_pixels=True)
        transfer_syntax_uid = dicom_file.file_meta.TransferSyntaxUID
        print(f"📂 File: {dicom_path}")
        print(f"🔍 Transfer Syntax UID: {transfer_syntax_uid}")

        encoding_type = ENCODING_DICT.get(str(transfer_syntax_uid), "🔴 Không xác định")
        print(f"✅ Kiểu mã hóa: {encoding_type}")

        return str(transfer_syntax_uid), encoding_type

    except Exception as e:
        print(f"❌ Lỗi khi đọc file DICOM: {str(e)}")
        return None, None


def read_dicom_metadata(dicom_path: str) -> Dict:
    """
    Read all metadata from a DICOM file.

    Args:
        dicom_path (str): Path to the DICOM file

    Returns:
        Dict: Dictionary containing all metadata from the DICOM file
    """
    try:
        dicom_file = pydicom.dcmread(dicom_path)

        # # Lấy các thông số quan trọng
        # metadata = {
        #     "File Name": dicom_path,
        #     "Patient Name": dicom_file.get("PatientName", "N/A"),
        #     "Patient ID": dicom_file.get("PatientID", "N/A"),
        #     "Patient Age": dicom_file.get("PatientAge", "N/A"),
        #     "Patient Sex": dicom_file.get("PatientSex", "N/A"),
        #     "Study Date": dicom_file.get("StudyDate", "N/A"),
        #     "Modality": dicom_file.get("Modality", "N/A"),
        #     "Institution Name": dicom_file.get("InstitutionName", "N/A"),
        #     "Manufacturer": dicom_file.get("Manufacturer", "N/A"),
        #     "Model": dicom_file.get("ManufacturerModelName", "N/A"),
        #     "Slice Thickness": dicom_file.get("SliceThickness", "N/A"),
        #     "Pixel Spacing": dicom_file.get("PixelSpacing", "N/A"),
        #     "Image Dimensions": f"{dicom_file.Rows} x {dicom_file.Columns}",
        #     "Bits Stored": dicom_file.get("BitsStored", "N/A"),
        #     "Photometric Interpretation": dicom_file.get(
        #         "PhotometricInterpretation", "N/A"
        #     ),
        #     "Samples Per Pixel": dicom_file.get("SamplesPerPixel", "N/A"),
        #     "Transfer Syntax UID": dicom_file.file_meta.TransferSyntaxUID,
        # }

        metadata = {"File Path": dicom_path}

        for elem in dicom_file.iterall():
            tag_name = elem.keyword if elem.keyword else elem.tag
            value = str(elem.value)
            metadata[tag_name] = value

        return metadata

    except Exception as e:
        print(f"❌ Lỗi khi đọc file DICOM ({dicom_path}): {e}")
        return {"File Path": dicom_path, "Error": str(e)}


def show_dicom_image(dicom_path: str) -> None:
    """
    Display a DICOM image using matplotlib.

    Args:
        dicom_path (str): Path to the DICOM file
    """
    try:
        dicom_file = pydicom.dcmread(dicom_path)
        photometric_interpretation = dicom_file.PhotometricInterpretation
        image = dicom_file.pixel_array

        if photometric_interpretation == "YBR_FULL":
            image = np.dot(image, [0.299, 0.587, 0.114])

        plt.imshow(image, cmap="gray" if len(image.shape) == 2 else None)
        plt.axis("off")
        plt.title(os.path.basename(dicom_path))
        plt.show()
    except Exception as e:
        print(f"❌ Không thể hiển thị ảnh DICOM: {str(e)}")


def view_dicom_info(dicom_paths: List[str]) -> None:
    """
    Check and display information for a list of DICOM files.

    Args:
        dicom_paths (List[str]): List of paths to DICOM files
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


def process_dicom_folder(
    folder_path: str, output_excel: str = "metadata_Chung_20241014_predict.xlsx"
) -> Optional[pd.DataFrame]:
    """
    Process all DICOM files in a folder and save metadata to Excel.

    Args:
        folder_path (str): Path to the folder containing DICOM files
        output_excel (str): Name of the output Excel file

    Returns:
        Optional[pd.DataFrame]: DataFrame containing metadata if successful, None otherwise
    """
    dicom_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".dcm")
    ]

    if not dicom_files:
        print("❌ Không tìm thấy file DICOM nào trong thư mục.")
        return None

    print(f"🔍 Đang xử lý {len(dicom_files)} file DICOM...")

    metadata_list = [read_dicom_metadata(dicom_file) for dicom_file in dicom_files]
    df = pd.DataFrame(metadata_list)

    df.to_excel(output_excel, index=False)
    print(f"✅ Dữ liệu đã được lưu vào {output_excel}")

    return df


def compare_dicom_metadata(file1: str, file2: str) -> None:
    """
    Compare metadata between two DICOM files.

    Args:
        file1 (str): Path to first DICOM file
        file2 (str): Path to second DICOM file
    """
    ds1 = pydicom.dcmread(file1)
    ds2 = pydicom.dcmread(file2)

    print(f"🔍 So sánh metadata giữa: {file1} và {file2}\n")

    for elem in ds1.dir():
        val1 = getattr(ds1, elem, "N/A")
        val2 = getattr(ds2, elem, "N/A")

        if val1 != val2:
            print(f"⚠️ Khác biệt: {elem}")
            print(f"   🔴 {file1}: {val1}")
            print(f"   🟢 {file2}: {val2}")
            print("-" * 50)


def fix_dicom_metadata(
    dicom_path: str,
    output_path: str,
    pixel_spacing: Optional[List[float]] = None,
    rescale_intercept: Optional[float] = None,
    photometric_interpretation: Optional[str] = None,
) -> None:
    """
    Fix various DICOM metadata issues.

    Args:
        dicom_path (str): Path to input DICOM file
        output_path (str): Path to save the fixed DICOM file
        pixel_spacing (Optional[List[float]]): New pixel spacing values
        rescale_intercept (Optional[float]): New rescale intercept value
        photometric_interpretation (Optional[str]): New photometric interpretation value
    """
    ds = pydicom.dcmread(dicom_path)

    if pixel_spacing is not None:
        ds.PixelSpacing = pixel_spacing
    if rescale_intercept is not None:
        ds.RescaleIntercept = rescale_intercept
    if photometric_interpretation is not None:
        ds.PhotometricInterpretation = photometric_interpretation

    ds.save_as(output_path, write_like_original=False)
    print(f"✅ Updated DICOM metadata and saved: {output_path}")


def main():
    """
    Main function to demonstrate usage of the DICOM utilities.
    """
    # Example usage
    dicom_folder = r"D:\Work\Clients\Job\dicom-sybil\Sybil\visualizations\serie_0"
    metadata_df = process_dicom_folder(dicom_folder)

    # Example of comparing two DICOM files
    link_oke = (
        r"D:\Work\Clients\Job\dicom-sybil\Sybil\custom\Chung_20241014\slice_130.dcm"
    )
    link_err = (
        r"D:\Work\Clients\Job\dicom-sybil\Sybil\visualizations\serie_0\slice_170.dcm"
    )
    compare_dicom_metadata(link_oke, link_err)

    # Example of fixing DICOM metadata
    fix_dicom_metadata(
        link_err,
        "output_fixed.dcm",
        pixel_spacing=[0.644531, 0.644531],
        rescale_intercept=-1024,
        photometric_interpretation="MONOCHROME2",
    )


if __name__ == "__main__":
    main()
