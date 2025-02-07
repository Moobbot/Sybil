import os
import imageio
import numpy as np
import pydicom
import torch
import torch.nn.functional as F
from sybil.serie import Serie
from typing import Dict, List, Union


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


def save_attention_images_dicom(
    overlayed_images: List[np.ndarray],
    cur_attention: np.ndarray,
    save_path: str,
    attention_threshold: float,
    dicom_metadata_list: List[pydicom.Dataset],
):
    """
    Lưu ảnh overlay có attention vượt ngưỡng dưới dạng DICOM màu (RGB).

    Args:
        overlayed_images (List[np.ndarray]): Danh sách ảnh đã overlay attention.
        cur_attention (np.ndarray): Attention values tương ứng với ảnh.
        save_path (str): Đường dẫn để lưu ảnh.
        attention_threshold (float): Ngưỡng attention để quyết định lưu ảnh.
        dicom_metadata_list (List[pydicom.Dataset]): Danh sách metadata của từng ảnh DICOM.

    Returns:
        None
    """
    os.makedirs(save_path, exist_ok=True)

    for idx, (img, attention) in enumerate(zip(overlayed_images, cur_attention)):
        if np.max(attention) > attention_threshold:
            dicom_path = os.path.join(save_path, f"slice_{idx}.dcm")

            try:
                # Lấy metadata từ ảnh gốc
                ds = dicom_metadata_list[idx].copy()

                # Chuyển ảnh overlay thành RGB (nếu chưa đúng định dạng)
                if img.shape[-1] != 3:
                    raise ValueError(f"Ảnh overlay không phải RGB: {img.shape}")

                # Chuyển đổi ảnh sang uint8 (0-255)
                img_uint8 = np.clip(img, 0, 255).astype(np.uint8)

                # Cấu hình metadata cho DICOM màu
                ds.Rows, ds.Columns, ds.SamplesPerPixel = img_uint8.shape
                ds.PhotometricInterpretation = "RGB"
                ds.BitsAllocated = 8
                ds.BitsStored = 8
                ds.HighBit = 7
                ds.PlanarConfiguration = 0  # Lưu ảnh theo thứ tự RGBRGB...
                ds.PixelRepresentation = 0

                # Kiểm tra và sao chép các metadata quan trọng
                for attr in ["PixelSpacing", "RescaleSlope", "RescaleIntercept"]:
                    if hasattr(dicom_metadata_list[idx], attr):
                        setattr(ds, attr, getattr(dicom_metadata_list[idx], attr))

                # Gán ảnh vào PixelData
                ds.PixelData = img_uint8.tobytes()

                # Lưu ảnh DICOM
                ds.save_as(dicom_path)
                print(f"✅ Saved DICOM overlay (RGB): {dicom_path}")

            except Exception as e:
                print(f"⚠️ Error saving DICOM slice {idx}: {str(e)}")
