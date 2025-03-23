import logging
import os
import imageio
import numpy as np
import pydicom
import torch
import torch.nn.functional as F
from sybil.serie import Serie
from typing import Dict, List, Union
import os

EPS = 1e-9  # 1e-3


def collate_attentions(
    attention_dict: Dict[str, np.ndarray], N: int, eps=EPS
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
    images: List[np.ndarray], attention: np.ndarray, gain: int = 3
):
    """
    Build overlayed images from a list of images and an attention map.

    Args:
        images (List[np.ndarray]): List of NumPy arrays representing the images.
        attention (np.ndarray): NumPy array containing attention maps.
    """
    overlayed_images = []
    N = len(images)
    for i in range(N):
        overlayed = np.zeros((512, 512, 3))
        overlayed[..., 2] = images[i]
        overlayed[..., 1] = images[i]
        overlayed[..., 0] = np.clip(
            (attention[i, ...] * gain * 256) + images[i],
            a_min=0,
            a_max=255,
        )

        overlayed_images.append(np.uint8(overlayed))

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

    Parameters:
    - overlayed_images: List of NumPy arrays representing the overlayed images.
    - cur_attention: List of NumPy arrays containing attention maps.
    - save_path: Path to save the generated files.
    - attention_threshold: Minimum threshold to determine if the attention map should be saved.
    - save_as_dicom: If True, saves as DICOM format, otherwise saves as PNG.
    - dicom_metadata_list: List of pydicom.Dataset objects containing original DICOM metadata (required if save_as_dicom is True).
    - input_files: List of original input file paths to get original filenames.
    """
    os.makedirs(save_path, exist_ok=True)

    # Validate DICOM metadata requirements upfront
    if save_as_dicom:
        if dicom_metadata_list is None:
            raise ValueError(
                "dicom_metadata_list is required when saving as DICOM format"
            )
        if len(dicom_metadata_list) != len(overlayed_images):
            logging.warning(
                "Mismatch: Number of DICOM metadata does not match the number of images!"
            )

    for idx, (img, attention) in enumerate(zip(overlayed_images, cur_attention)):
        if np.mean(attention) > attention_threshold:
            # Get original filename if available, otherwise use index
            if input_files and idx < len(input_files):
                original_filename = os.path.basename(input_files[idx])
                # Remove extension and add pred_ prefix
                base_filename = f"pred_{os.path.splitext(original_filename)[0]}"
            else:
                base_filename = f"pred_{idx}"

            if not save_as_dicom:
                # Save as PNG
                png_path = os.path.join(save_path, f"{base_filename}.png")
                imageio.imwrite(png_path, img)
                print(f"Saved overlay PNG: {png_path}")
            else:
                if idx >= len(dicom_metadata_list):
                    logging.warning(
                        f"Skipping slice {idx}: No corresponding DICOM metadata found."
                    )
                    continue

                try:
                    # Get metadata from original image
                    ds = dicom_metadata_list[idx].copy()

                    # Convert image to uint8 (0-255)
                    img_uint8 = np.clip(img, 0, 255).astype(np.uint8)

                    # Configure metadata for RGB DICOM
                    ds.Rows, ds.Columns = img_uint8.shape[:2]
                    ds.SamplesPerPixel = 3
                    ds.PhotometricInterpretation = "RGB"
                    ds.BitsAllocated = 8
                    ds.BitsStored = 8
                    ds.HighBit = 7
                    ds.PlanarConfiguration = 0  # RGB contiguous
                    ds.PixelRepresentation = 0

                    # Make sure rescale values are set properly
                    ds.RescaleIntercept = 0
                    ds.RescaleSlope = 1

                    # Set VOI LUT function to LINEAR for proper scaling
                    ds.VOILUTFunction = "LINEAR"

                    # Copy important metadata
                    for attr in [
                        "PixelSpacing",
                        "SliceLocation",
                        "ImagePositionPatient",
                        "ImageOrientationPatient",
                        "InstanceNumber",
                    ]:
                        if hasattr(dicom_metadata_list[idx], attr):
                            setattr(ds, attr, getattr(dicom_metadata_list[idx], attr))

                    # Assign image to PixelData
                    ds.PixelData = img_uint8.tobytes()

                    # Save DICOM image
                    dicom_path = os.path.join(save_path, f"{base_filename}.dcm")
                    ds.save_as(dicom_path)
                    logging.info(f"✅ Successfully saved DICOM overlay: {dicom_path}")
                    print(f"✅ Saved DICOM overlay (RGB): {dicom_path}")

                except Exception as e:
                    logging.error(f"⚠️ Error saving DICOM slice {idx}: {str(e)}")
                    print(f"⚠️ Error saving DICOM slice {idx}: {str(e)}")


def visualize_attentions(
    series: Union[Serie, List[Serie]],
    attentions: List[Dict[str, np.ndarray]],
    save_directory: str = None,
    gain: int = 3,
    attention_threshold: float = EPS,
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

            # Save images in the correct order
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
    eps: float = EPS,
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
    """
    # Calculate the attention map
    attention = collate_attentions(attention_dict, N, eps)

    # Calculate the attention score for each image
    attention_scores = []
    for i in range(N):
        score = np.mean(attention[i])  # Can change the way to calculate the score
        attention_scores.append((i, score))

    # Sort by score in descending order
    ranked_indices = sorted(attention_scores, key=lambda x: x[1], reverse=True)

    # Create a list of dictionaries with the following keys:
    # - rank: The rank of the image
    # - attention_score: The attention score of the image
    # - image: The image
    # - attention_map: The attention map of the image
    ranked_images = []
    for rank, (idx, score) in enumerate(ranked_indices, 1):
        ranked_images.append(
            {
                "rank": rank,
                "attention_score": float(score),
                "image": images[idx],
                "attention_map": attention[idx],
            }
        )

    return ranked_images
