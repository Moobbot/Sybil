import os
from typing import Dict, List, Union

import cv2
import imageio
import numpy as np
import pydicom
import torch
import torch.nn.functional as F
from PIL import Image
from pydicom.uid import ExplicitVRLittleEndian

from sybil.serie import Serie


def collate_attentions(
    attention_dict: Dict[str, np.ndarray], N: int, eps=1e-6
) -> np.ndarray:
    """Tổng hợp attention từ mô hình."""
    if not attention_dict:
        raise ValueError(
            "⚠️ Attention dictionary is empty. Ensure model returns valid attention maps."
        )

    a1 = torch.tensor(attention_dict.get("image_attention_1"), dtype=torch.float32)
    v1 = torch.tensor(attention_dict.get("volume_attention_1"), dtype=torch.float32)

    # Mean over ensemble
    a1 = torch.exp(a1).mean(0)
    v1 = torch.exp(v1).mean(0)

    attention = a1 * v1.unsqueeze(-1)
    attention = attention.view(1, 25, 16, 16)

    # Upscale attention map
    attention_up = (
        F.interpolate(attention.unsqueeze(0), (N, 512, 512), mode="trilinear")
        .cpu()
        .numpy()
        .squeeze()
    )
    attention_up[attention_up <= eps] = 0.0  # Apply threshold

    return attention_up


def build_overlayed_images(
    images: List[np.ndarray], attention: np.ndarray, gain: int = 3
) -> List[np.ndarray]:
    """Xây dựng ảnh overlay từ attention maps."""
    overlayed_images = []
    for img, att in zip(images, attention):
        overlayed = np.zeros((512, 512, 3), dtype=np.uint8)
        overlayed[..., 2] = img  # Channel Red
        overlayed[..., 1] = img  # Channel Green
        overlayed[..., 0] = np.clip(
            (att * gain * 256) + img, a_min=0, a_max=255
        )  # Channel Blue

        overlayed_images.append(overlayed)

    return overlayed_images


def visualize_attentions(
    series: Union[Serie, List[Serie]],
    attentions: List[Dict[str, np.ndarray]],
    save_directory: str = None,
    gain: int = 3,
    attention_threshold: float = 1e-3,
    save_as_dicom: bool = False,  # Lựa chọn lưu DICOM hoặc PNG
    dicom_metadata_list: List[pydicom.Dataset] = None,  # Danh sách metadata DICOM
) -> List[List[np.ndarray]]:
    """
    Tạo ảnh overlay từ attention và lưu vào thư mục.
    Args:
        series (Serie): series object
        attentions (Dict[str, np.ndarray]): attention dictionary output from model
        save_directory (str, optional): where to save the images. Defaults to None.
        gain (int, optional): how much to scale attention values by for visualization. Defaults to 3.
        attention_threshold (float, optional): Minimum attention value to consider saving an image. Defaults to 1e-3.
        save_as_dicom (bool): Nếu True, lưu ảnh dưới dạng DICOM thay vì PNG.
        dicom_metadata_list (List[pydicom.Dataset], optional): Danh sách metadata của từng ảnh DICOM.

    Returns:
        List[List[np.ndarray]]: list of list of overlayed images
    """
    print("=== visualize_attentions ===")

    if isinstance(series, Serie):
        series = [series]

    if not attentions or len(attentions) == 0:
        raise ValueError(
            "⚠️ Attention data is empty. Ensure `return_attentions=True` when predicting."
        )

    series_overlays = []
    for serie_idx, serie in enumerate(series):
        images = serie.get_raw_images()
        N = len(images)

        if serie_idx >= len(attentions) or attentions[serie_idx] is None:
            print(f"⚠️ Warning: Missing attention data for series {serie_idx}")
            continue

        cur_attention = collate_attentions(attentions[serie_idx], N)
        overlayed_images = build_overlayed_images(images, cur_attention, gain)

        if save_directory:
            save_path = os.path.join(save_directory, f"serie_{serie_idx}")
            os.makedirs(save_path, exist_ok=True)

            # Kiểm tra lựa chọn lưu ảnh PNG hoặc DICOM
            if save_as_dicom and dicom_metadata_list:
                save_attention_images_dicom(
                    overlayed_images,
                    cur_attention,
                    save_path,
                    attention_threshold,
                    dicom_metadata_list,
                )
            else:
                save_attention_images(
                    overlayed_images, cur_attention, save_path, attention_threshold
                )

            save_images(overlayed_images, save_path, f"serie_{serie_idx}")

        series_overlays.append(overlayed_images)

    return series_overlays


def save_images(img_list: List[np.ndarray], directory: str, name: str):
    """
    Saves a list of images as a GIF in the specified directory with the given name.

    Args:
        ``img_list`` (List[np.ndarray]): A list of numpy arrays representing the images to be saved.
        ``directory`` (str): The directory where the GIF should be saved.
        ``name`` (str): The name of the GIF file.

    Returns:
        None
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{name}.gif")
    imageio.mimsave(path, img_list)
    print(f"GIF saved to: {path}")


def save_attention_images(
    overlayed_images: List[np.ndarray],
    cur_attention: np.ndarray,
    save_path: str,
    attention_threshold: float,
):
    """
    Lưu ảnh overlay dạng PNG nếu attention vượt ngưỡng.

    Args:
        overlayed_images (List[np.ndarray]): Danh sách ảnh đã overlay attention.
        cur_attention (np.ndarray): Attention values tương ứng với ảnh.
        save_path (str): Đường dẫn để lưu ảnh.
        attention_threshold (float): Ngưỡng attention để quyết định lưu ảnh.

    Returns:
        None
    """
    os.makedirs(save_path, exist_ok=True)
    for idx, (img, attention) in enumerate(zip(overlayed_images, cur_attention)):
        if np.max(attention) > attention_threshold:
            overlay_path = os.path.join(save_path, f"slice_{idx}.png")
            imageio.imwrite(overlay_path, img)
            print(f"Saved overlay PNG: {overlay_path}")


from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid


def convert_png_to_dicom(
    png_path: str, dicom_metadata: pydicom.Dataset, output_dicom_path: str
):
    """
    Chuyển đổi PNG overlay sang DICOM nhưng giữ RGB và điều chỉnh Window Level.

    Args:
        png_path (str): Đường dẫn ảnh PNG overlay.
        dicom_metadata (pydicom.Dataset): Metadata DICOM gốc.
        output_dicom_path (str): Đường dẫn lưu file DICOM.

    Returns:
        None
    """
    try:
        # Đọc ảnh PNG (là ảnh overlay đã chèn)
        img = Image.open(png_path).convert("RGB")  # Giữ RGB
        img_array = np.array(img)

        # Đảm bảo dữ liệu DICOM ở dạng 8-bit
        img_rgb = np.clip(img_array, 0, 255).astype(np.uint8)

        # ✅ Tạo file meta cho DICOM P10
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = dicom_metadata.SOPClassUID
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian  # Không nén
        file_meta.ImplementationClassUID = generate_uid()

        # ✅ Sao chép metadata từ file DICOM gốc
        ds = Dataset()
        ds.update(dicom_metadata)
        ds.file_meta = file_meta  # Gán metadata chuẩn DICOM P10

        # ✅ Giữ RGB
        ds.Rows = img_rgb.shape[0]
        ds.Columns = img_rgb.shape[1]
        ds.SamplesPerPixel = 3  # Giữ RGB
        ds.PhotometricInterpretation = "RGB"
        ds.BitsAllocated = 8  # RGB sử dụng 8-bit
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PlanarConfiguration = 0  # RGBRGBRGB...
        ds.PixelRepresentation = 0  # Unsigned

        # ✅ Chỉnh lại Window Center & Window Width để phù hợp với RGB
        ds.WindowCenter = [40, 40, 40]  # Dựa trên file DICOM hiển thị tốt
        ds.WindowWidth = [250, 250, 250]

        # ✅ Gán ảnh vào PixelData
        ds.PixelData = img_rgb.tobytes()

        # ✅ Lưu file DICOM P10 với định dạng chuẩn
        ds.save_as(output_dicom_path, write_like_original=False)
        print(f"✅ Converted PNG to DICOM (RGB, Fixed WL/WW): {output_dicom_path}")

    except Exception as e:
        print(f"⚠️ Error converting PNG to DICOM: {str(e)}")


def save_attention_images_dicom(
    overlayed_images: List[np.ndarray],
    cur_attention: np.ndarray,
    save_path: str,
    attention_threshold: float,
    dicom_metadata_list: List[pydicom.Dataset],
):
    """
    Lưu ảnh overlay dưới dạng PNG, sau đó convert sang DICOM RGB.

    Args:
        overlayed_images (List[np.ndarray]): Danh sách ảnh overlay.
        cur_attention (np.ndarray): Attention map.
        save_path (str): Đường dẫn lưu file.
        attention_threshold (float): Ngưỡng attention để lưu ảnh.
        dicom_metadata_list (List[pydicom.Dataset]): Metadata DICOM.

    Returns:
        None
    """
    os.makedirs(save_path, exist_ok=True)

    for idx, (img, attention) in enumerate(zip(overlayed_images, cur_attention)):
        if np.max(attention) > attention_threshold:
            png_path = os.path.join(save_path, f"slice_{idx}.png")
            dicom_path = os.path.join(save_path, f"slice_{idx}.dcm")

            try:
                # Lưu ảnh PNG trước
                cv2.imwrite(png_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                print(f"✅ Saved PNG: {png_path}")

                # Convert PNG sang DICOM
                convert_png_to_dicom(png_path, dicom_metadata_list[idx], dicom_path)

            except Exception as e:
                print(f"⚠️ Error saving DICOM slice {idx}: {str(e)}")
