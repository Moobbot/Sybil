import logging
import os
import imageio
import numpy as np
import pydicom
import torch
import torch.nn.functional as F
from sybil.serie import Serie
from typing import Dict, List, Union
from .config import VISUALIZATION_CONFIG as cfg
from .dicom_handler import DicomHandler


def collate_attentions(
    attention_dict: Dict[str, np.ndarray], N: int, eps=cfg['EPS']
) -> np.ndarray:
    """
    Collate attention maps from a dictionary of attention maps.

    Args:
        attention_dict (Dict[str, np.ndarray]): Dictionary containing attention maps.
        N (int): Number of images.
    """
    a1 = attention_dict["image_attention_1"]
    v1 = attention_dict["volume_attention_1"]

    a1 = torch.Tensor(a1)
    v1 = torch.Tensor(v1)

    # take mean attention over ensemble
    a1 = torch.exp(a1).mean(0)
    v1 = torch.exp(v1).mean(0)

    attention = a1 * v1.unsqueeze(-1)
    attention = attention.view(1, 25, 16, 16)

    attention_up = F.interpolate(
        attention.unsqueeze(0), (N, 512, 512), mode="trilinear"
    )
    attention_up = attention_up.cpu().numpy()
    attention_up = attention_up.squeeze()
    if eps:
        attention_up[attention_up <= eps] = 0.0

    return attention_up


def build_overlayed_images(
    images: List[np.ndarray], 
    attention: np.ndarray, 
    gain: int = cfg['DEFAULT_GAIN']
) -> List[np.ndarray]:
    """
    Build overlayed images from a list of images and an attention map.
    If attention values are below threshold, returns original image without overlay.

    Args:
        images (List[np.ndarray]): List of NumPy arrays representing the images.
        attention (np.ndarray): NumPy array containing attention maps.
        gain (int): Factor to scale attention values.
    """
    overlayed_images = []
    N = len(images)
    for i in range(N):
        # Kiểm tra nếu attention map có giá trị đáng kể
        if np.any(attention[i] > cfg['EPS']):
            # Tạo overlay nếu có attention đáng kể
            overlayed = np.zeros((512, 512, 3))
            overlayed[..., 2] = images[i]
            overlayed[..., 1] = images[i]
            overlayed[..., 0] = np.clip(
                (attention[i, ...] * gain * 256) + images[i],
                a_min=0,
                a_max=255,
            )
            overlayed_images.append(np.uint8(overlayed))
        else:
            # Nếu không có attention đáng kể, sử dụng ảnh gốc
            # Chuyển ảnh gốc thành RGB
            original = np.zeros((512, 512, 3))
            original[..., 0] = images[i]
            original[..., 1] = images[i]
            original[..., 2] = images[i]
            overlayed_images.append(np.uint8(original))

    return overlayed_images


def save_gif(img_list: List[np.ndarray], directory: str, name: str):
    """
    Saves a list of images as a GIF in the specified directory with the given name.

    Args:
        ``img_list`` (List[np.ndarray]): A list of numpy arrays representing the images to be saved.
        ``directory`` (str): The directory where the GIF should be saved.
        ``name`` (str): The name of the GIF file.

    Returns:
        None
    """
    import imageio

    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{name}.gif")
    imageio.mimsave(path, img_list)
    print(f"GIF saved to: {path}")


def remove_diacritics(text):
    """Remove diacritics from Vietnamese text"""
    import unicodedata

    text = unicodedata.normalize("NFKD", text)
    return "".join([c for c in text if not unicodedata.combining(c)])


def save_attention_images(
    overlayed_images: List[np.ndarray],
    cur_attention: np.ndarray,
    save_path: str,
    attention_threshold: float,
    save_as_dicom: bool = False,
    dicom_metadata_list: List[pydicom.Dataset] = None,
    input_files: List[str] = None,
):
    """
    Saves overlayed attention images as PNG or DICOM files.
    """
    os.makedirs(save_path, exist_ok=True)

    if save_as_dicom and dicom_metadata_list is None:
        raise ValueError("dicom_metadata_list is required when saving as DICOM format")

    N = len(overlayed_images)
    indices = list(range(N))
    num_digits = len(str(N))

    for i in indices:
        attention = cur_attention[i]
        mean_attention = np.mean(attention)
        
        # Get patient name
        patient_name = ""
        if input_files:
            base_name = os.path.splitext(os.path.basename(input_files[i]))[0]
            parts = base_name.split('_')
            if parts and parts[-1].isdigit():
                patient_name = '_'.join(parts[:-1])
            else:
                patient_name = base_name
        else:
            patient_name = f"Unknown_Patient"

        base_filename = f"pred_{patient_name}_{(N-1)-i:0{num_digits}d}"

        # Thêm thông tin về loại ảnh vào tên file
        if mean_attention <= cfg['EPS']:
            base_filename = f"{base_filename}_original"
            logging.info(f"Saving original image for slice {i} (no significant attention)")
        
        if not save_as_dicom:
            # Save as PNG
            png_path = os.path.join(save_path, f"{base_filename}.png")
            imageio.imwrite(png_path, overlayed_images[i])
            print(f"Saved {'original' if mean_attention <= cfg['EPS'] else 'overlay'} PNG: {png_path}")
        else:
            if i >= len(dicom_metadata_list):
                logging.warning(
                    f"Skipping slice {i}: No corresponding DICOM metadata found."
                )
                continue

            DicomHandler.save_overlay_as_dicom(
                overlayed_images[i],
                dicom_metadata_list[i],
                save_path,
                base_filename
            )


def visualize_attentions(
    series: Union[Serie, List[Serie]],
    attentions: List[Dict[str, np.ndarray]],
    save_directory: str = None,
    gain: int = cfg['DEFAULT_GAIN'],
    attention_threshold: float = cfg['DEFAULT_ATTENTION_THRESHOLD'],
    save_as_dicom: bool = False,
    dicom_metadata_list: List[pydicom.Dataset] = None,
    input_files: List[str] = None,
) -> List[List[np.ndarray]]:
    """
    Generates overlayed attention images and saves them as PNG or DICOM.

    Args:
        series (Union[Serie, List[Serie]]): The series object(s) containing images.
        attentions (List[Dict[str, np.ndarray]]): A list of attention maps per series.
        save_directory (Optional[str]): Directory to save the images. Defaults to None.
        gain (int): Factor to scale attention values for visualization. Defaults to 3.
        attention_threshold (float): Minimum attention value to consider saving an image.
        save_as_dicom (bool): If True, saves images as DICOM instead of PNG.
        dicom_metadata_list (Optional[List[pydicom.Dataset]]): Metadata list for DICOM images.
        input_files (Optional[List[str]]): List of original input file paths.

    Returns:
        List[List[np.ndarray]]: List of overlayed image lists per series.
    """
    print("=== visualize_attentions ===")

    if isinstance(series, Serie):
        series = [series]

    if not attentions or len(attentions) == 0:
        raise ValueError(
            "⚠️ Attention data is empty. Ensure `return_attentions=True` when predicting."
        )

    if dicom_metadata_list and len(dicom_metadata_list) < len(series):
        logging.warning(
            "⚠️ DICOM metadata list is shorter than the number of series. Some images may be missing metadata."
        )

    series_overlays = []
    for serie_idx, serie in enumerate(series):
        images = serie.get_raw_images()
        N = len(images)

        if serie_idx >= len(attentions) or attentions[serie_idx] is None:
            print(f"⚠️ Warning: Missing attention data for series {serie_idx}")
            logging.warning(
                f"⚠️ Missing attention data for series {serie_idx}. Skipping."
            )
            continue

        cur_attention = collate_attentions(attentions[serie_idx], N)

        overlayed_images = build_overlayed_images(images, cur_attention, gain)

        if save_directory:
            save_path = os.path.join(save_directory, f"serie_{serie_idx}")
            os.makedirs(save_path, exist_ok=True)

            save_attention_images(
                overlayed_images,
                cur_attention,
                save_path,
                attention_threshold,
                save_as_dicom,
                dicom_metadata_list,
                input_files,
            )

            # Save GIF with correctly ordered images
            save_gif(overlayed_images, save_path, f"serie_{serie_idx}")

        series_overlays.append(overlayed_images)

    return series_overlays


def rank_images_by_attention(
    attention_dict: Dict[str, np.ndarray],
    images: List[np.ndarray],
    N: int,
    eps: float = cfg['EPS'],
) -> List[Dict[str, Union[int, float, np.ndarray]]]:
    """
    Rank images based on the predicted attention score.

    Args:
        attention_dict (Dict[str, np.ndarray]): Dictionary containing attention maps
        images (List[np.ndarray]): List of original images
        N (int): Number of images
        eps (float): Minimum threshold for attention score

    Returns:
        List[Dict[str, Union[int, float, np.ndarray]]]: List of ranked images, each element is a dict containing:
            - rank: The rank of the image
            - attention_score: The attention score
            - image: Original image
            - attention_map: Corresponding attention map
            - original_index: Original index of the image
    """
    # Calculate the attention map
    attention = collate_attentions(attention_dict, N, eps)

    # Calculate the attention score for each image
    attention_scores = []
    for i in range(N):
        # Tính toán attention score cho từng slice
        slice_attention = attention[i]
        # Lấy trung bình của các giá trị attention > eps
        mask = slice_attention > eps
        if mask.any():
            score = slice_attention[mask].mean()
        else:
            score = 0.0
        attention_scores.append((i, score))

    # Sort by score in descending order
    # ranked_indices = sorted(attention_scores, key=lambda x: x[1], reverse=True)

    # Create a list of dictionaries with ranking info
    ranked_images = []
    for rank, (idx, score) in enumerate(attention_scores, 1):
        ranked_images.append(
            {
                "rank": rank,
                "attention_score": float(score),
                "image": images[idx],
                "attention_map": attention[idx],
                "original_index": idx,
            }
        )

    return ranked_images
