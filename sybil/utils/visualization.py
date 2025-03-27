import logging
import os
import imageio
import numpy as np
import pydicom
import torch
import torch.nn.functional as F
from sybil.serie import Serie
from typing import Dict, List, Union
from config import MODEL_CONFIG, VISUALIZATION_CONFIG as cfg
from .config import VISUALIZATION_CONFIG as viz_cfg
from .dicom_handler import DicomHandler


def collate_attentions(
    attention_dict: Dict[str, np.ndarray], N: int, eps=viz_cfg["EPS"]
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
    gain: int = viz_cfg["DEFAULT_GAIN"],
    save_original: bool = False,
) -> List[np.ndarray]:
    """
    Build overlayed images from a list of images and an attention map.

    Args:
        images (List[np.ndarray]): List of NumPy arrays representing the images.
        attention (np.ndarray): NumPy array containing attention maps.
        gain (int): Factor to scale attention values.
        save_original (bool): If True, save original image when no significant attention.
                            If False, always create overlay even with minimal attention.
    """
    overlayed_images = []
    N = len(images)
    for i in range(N):
        if not save_original or np.any(attention[i] > viz_cfg["EPS"]):
            # Tạo overlay cho mọi trường hợp nếu save_original=False
            # hoặc khi có attention đáng kể nếu save_original=True
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
            # Chỉ lưu ảnh gốc khi save_original=True và không có attention đáng kể
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
    save_as_dicom: bool = MODEL_CONFIG["SAVE_AS_DICOM_DEFAULT"],
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
            parts = base_name.split("_")
            if parts and parts[-1].isdigit():
                patient_name = "_".join(parts[:-1])
            else:
                patient_name = base_name
        else:
            patient_name = f"Unknown_Patient"

        base_filename = f"pred_{patient_name}_{(N-1)-i:0{num_digits}d}"

        # Thêm thông tin về loại ảnh vào tên file
        if mean_attention <= viz_cfg["EPS"]:
            base_filename = (
                f"{base_filename}{viz_cfg['FILE_NAMING']['ORIGINAL_SUFFIX']}"
            )
            logging.info(
                f"Saving original image for slice {i} (no significant attention)"
            )

        if not save_as_dicom:
            # Save as PNG
            png_path = os.path.join(save_path, f"{base_filename}.png")
            imageio.imwrite(png_path, overlayed_images[i])
            print(
                f"Saved {'original' if mean_attention <= viz_cfg['EPS'] else 'overlay'} PNG: {png_path}"
            )
        else:
            if i >= len(dicom_metadata_list):
                logging.warning(
                    f"Skipping slice {i}: No corresponding DICOM metadata found."
                )
                continue

            DicomHandler.save_overlay_as_dicom(
                overlayed_images[i], dicom_metadata_list[i], save_path, base_filename
            )


def visualize_attentions(
    series: Union[Serie, List[Serie]],
    attentions: List[Dict[str, np.ndarray]],
    save_directory: str = None,
    gain: int = viz_cfg["DEFAULT_GAIN"],
    save_as_dicom: bool = MODEL_CONFIG["SAVE_AS_DICOM_DEFAULT"],
    save_original: bool = MODEL_CONFIG["SAVE_ORIGINAL_DEFAULT"],
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
        save_original (bool): If True, save original image when no significant attention.
                            If False, always create overlay. Defaults to False.

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

        overlayed_images = build_overlayed_images(
            images, cur_attention, gain, save_original
        )

        if save_directory:
            save_path = os.path.join(save_directory, f"serie_{serie_idx}")
            os.makedirs(save_path, exist_ok=True)

            save_attention_images(
                overlayed_images,
                cur_attention,
                save_path,
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
    eps: float = viz_cfg["EPS"],
    return_type: str = cfg["RANKING"]["DEFAULT_RETURN_TYPE"],
    top_k: int = cfg["RANKING"]["DEFAULT_TOP_K"],
) -> Union[List[Dict[str, Union[int, float, np.ndarray]]], None]:
    """
    Rank images based on the predicted attention score with emphasis on intensity.

    Args:
        attention_dict (Dict[str, np.ndarray]): Dictionary containing attention maps
        images (List[np.ndarray]): List of original images
        N (int): Number of images
        eps (float): Minimum threshold for attention score
        return_type (str): Type of return:
            - 'all': Return all images (default)
            - 'top': Return top K images
            - 'none': Don't return images
        top_k (int): Number of top images to return when return_type='top'

    Returns:
        Union[List[Dict[str, Union[int, float, np.ndarray]]], None]: List of ranked images or None
        Each element in list contains:
            - attention_score: The attention score
            - original_index: Original index of the image
    """
    # Calculate the attention map
    attention = collate_attentions(attention_dict, N, eps)

    # Calculate the attention score for each image
    attention_scores = []
    for i in range(N):
        slice_attention = attention[i]
        mask = slice_attention > eps

        if mask.any():
            # Calculate key metrics
            score = slice_attention[mask].mean()
            # Store diagnostic information for debugging
            debug_info = {
                "score": float(score),
            }

            attention_scores.append((i, score, debug_info))
        else:
            attention_scores.append(
                (
                    i,
                    0.0,
                    {"score": 0},
                )
            )

    # Sort by score in descending order
    sorted_scores = sorted(attention_scores, key=lambda x: x[1], reverse=True)

    # Print debug information for top images to help with tuning
    print(f"\nAttention Score Ranking (Top {top_k}):")
    for rank, (idx, score, debug) in enumerate(
        sorted_scores[: min(top_k, len(sorted_scores))], 1
    ):
        print(f"Rank {rank}: Index {idx}, Score: {score:.6f}")
        print(f"  Score: {debug['score']:.6f}")

    # Return None if return_type is 'none'
    if return_type.lower() == "none":
        return None

    # Create output list with appropriate number of images
    ranked_images = []
    scores_to_process = (
        sorted_scores[:top_k] if return_type.lower() == "top" else sorted_scores
    )

    for rank, (idx, score, _) in enumerate(scores_to_process, 1):
        ranked_images.append(
            {
                "attention_score": float(score),
                "original_index": idx,
            }
        )

    return ranked_images


def rank_images_by_attention_vip(
    attention_dict: Dict[str, np.ndarray],
    images: List[np.ndarray],
    N: int,
    eps: float = viz_cfg["EPS"],
    return_type: str = cfg["RANKING"]["DEFAULT_RETURN_TYPE"],
    top_k: int = cfg["RANKING"]["DEFAULT_TOP_K"],
) -> Union[List[Dict[str, Union[int, float, np.ndarray]]], None]:
    """
    Rank images based on the predicted attention score with emphasis on intensity.

    Args:
        attention_dict (Dict[str, np.ndarray]): Dictionary containing attention maps
        images (List[np.ndarray]): List of original images
        N (int): Number of images
        eps (float): Minimum threshold for attention score
        return_type (str): Type of return:
            - 'all': Return all images (default)
            - 'top': Return top K images
            - 'none': Don't return images
        top_k (int): Number of top images to return when return_type='top'

    Returns:
        Union[List[Dict[str, Union[int, float, np.ndarray]]], None]: List of ranked images or None
        Each element in list contains:
            - rank: The rank of the image (1-based)
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
        slice_attention = attention[i]
        mask = slice_attention > eps

        if mask.any():
            # Calculate key metrics
            max_attention = slice_attention.max()
            total_attention = slice_attention[mask].sum()
            mean_attention = slice_attention[mask].mean()
            area_ratio = mask.sum() / mask.size

            # Calculate high-intensity attention metrics
            if (
                mask.sum() >= 5
            ):  # Ensure we have enough pixels for percentile calculation
                p75_attention = np.percentile(slice_attention[mask], 75)
                p90_attention = np.percentile(slice_attention[mask], 90)
                p95_attention = np.percentile(slice_attention[mask], 95)

                # Count pixels in different intensity bands
                high_band = (slice_attention > p90_attention).sum()
                very_high_band = (slice_attention > p95_attention).sum()
            else:
                p75_attention = mean_attention
                p90_attention = max_attention
                p95_attention = max_attention
                high_band = mask.sum()
                very_high_band = mask.sum()

            # Prioritize maximum intensity with a power function to emphasize peaks
            intensity_score = max_attention**2

            # Calculate weighted score that heavily favors intensity over area
            # This formula gives much higher weight to maximum intensity and high percentiles
            intensity_weight = 0.8
            area_weight = 0.2

            # Intensity component emphasizes peak values
            intensity_component = (
                max_attention * 0.5 + p95_attention * 0.3 + p90_attention * 0.2
            ) * intensity_weight

            # Area component gives some consideration to total attention
            area_component = total_attention * area_weight

            # Add a bonus for very high intensity regions
            high_intensity_bonus = very_high_band * p95_attention * 0.1

            score = intensity_component + area_component + high_intensity_bonus

            # Store diagnostic information for debugging
            debug_info = {
                "max": float(max_attention),
                "mean": float(mean_attention),
                "p95": float(p95_attention) if mask.sum() >= 5 else 0,
                "area_ratio": float(area_ratio),
                "final_score": float(score),
            }

            attention_scores.append((i, score, debug_info))
        else:
            attention_scores.append(
                (
                    i,
                    0.0,
                    {"max": 0, "mean": 0, "p95": 0, "area_ratio": 0, "final_score": 0},
                )
            )

    # Sort by score in descending order
    sorted_scores = sorted(attention_scores, key=lambda x: x[1], reverse=True)

    # Print debug information for top images to help with tuning
    print(f"\nAttention Score Ranking (Top {top_k}):")
    for rank, (idx, score, debug) in enumerate(
        sorted_scores[: min(top_k, len(sorted_scores))], 1
    ):
        print(f"Rank {rank}: Index {idx}, Score: {score:.6f}")
        print(
            f"  Max: {debug['max']:.6f}, Mean: {debug['mean']:.6f}, P95: {debug['p95']:.6f}, Area: {debug['area_ratio']:.6f}"
        )

    # Return None if return_type is 'none'
    if return_type.lower() == "none":
        return None

    # Create output list with appropriate number of images
    ranked_images = []
    scores_to_process = (
        sorted_scores[:top_k] if return_type.lower() == "top" else sorted_scores
    )

    for rank, (idx, score, _) in enumerate(scores_to_process, 1):
        ranked_images.append(
            {
                "rank": rank,
                "attention_score": float(score),
                "image": images[idx],
                "original_index": idx,
            }
        )

    return ranked_images
