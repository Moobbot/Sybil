import json
import os
import pickle
import shutil
import tempfile
import typing
import urllib
import zipfile
from typing import Dict, Literal

import pydicom
from flask import logging

from config import (
    CALIBRATOR_PATH,
    CHECKPOINT_URL,
    FOLDERS,
    MODEL_CONFIG,
    MODEL_PATHS,
)
from config import VISUALIZATION_CONFIG as cfg
from input_series import detect_file_type, list_input_files
from sybil.datasets import utils as utils_datasets
from sybil.model import Sybil
from sybil.serie import Serie
from sybil.utils import logging_utils
from sybil.utils.config import VISUALIZATION_CONFIG as vsl_config
from sybil.utils.visualization import rank_images_by_attention, visualize_attentions


def download_checkpoints():
    """Download and extract checkpoint if not exist."""
    if not os.path.exists(FOLDERS["CHECKPOINT"]) or not all(
        os.path.exists(p) for p in MODEL_PATHS
    ):
        # P4c: checkpoints now live in a named volume (they survive container
        # re-creation). Download + extract into a TEMP directory in the same volume,
        # then os.replace each file: an interruption half-way leaves no TRUNCATED
        # .ckpt in the real location (otherwise the next start sees "present" ->
        # does not download again -> fails to load forever).
        ckpt_dir = FOLDERS["CHECKPOINT"]
        os.makedirs(ckpt_dir, exist_ok=True)
        tmp_dir = tempfile.mkdtemp(prefix=".download-", dir=ckpt_dir)
        try:
            print(f"Downloading checkpoints from {CHECKPOINT_URL}...")
            zip_path = os.path.join(tmp_dir, "sybil_checkpoints.zip")
            urllib.request.urlretrieve(CHECKPOINT_URL, zip_path)

            print("Extracting checkpoints...")
            extract_dir = os.path.join(tmp_dir, "extract")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
            for root, _, names in os.walk(extract_dir):
                for name in names:
                    src = os.path.join(root, name)
                    dst = os.path.join(ckpt_dir, os.path.relpath(src, extract_dir))
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    os.replace(src, dst)
            print("Checkpoints downloaded and extracted successfully.")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


# P4c: which load branch ran + exactly which weight files. routes.py uses it to
# build the version identity (model_info.py). Does not change the signature/return
# value of load_model (call_model.py calls it below).
LOAD_INFO = {"fallback": False, "weight_paths": []}


def load_model(model_name="sybil_ensemble"):
    """
    Load a trained Sybil model from a checkpoint or a directory of checkpoints.
    If there is no model or calibrator, download from CHECKPOINT_URL.

    Returns:
        Sybil: Model object has loaded.
    """
    # Check and download checkpoints if needed
    if not all(os.path.exists(p) for p in MODEL_PATHS) or not os.path.exists(
        CALIBRATOR_PATH
    ):
        print("Model and Calibrator checkpoints not found. Downloading...")
        download_checkpoints()

    print("Loading Sybil model...")
    try:
        model = Sybil(name_or_path=MODEL_PATHS, calibrator_path=CALIBRATOR_PATH)
        LOAD_INFO.update(fallback=False, weight_paths=[*MODEL_PATHS, CALIBRATOR_PATH])
    except Exception as e:
        # Previously a bare `except:` with no log: if the configured checkpoint failed
        # to load, it SILENTLY switched to other weights. That behaviour is kept (P4c
        # does not block clinical use) but it is RECORDED: result versions carry the
        # `fallback` flag.
        print(f"Could not load the configured checkpoint ({type(e).__name__}: {e}) -> using '{model_name}'")
        model = Sybil(model_name)
        LOAD_INFO.update(fallback=True, weight_paths=[])
    print("Model loaded successfully.")
    return model


def get_input_files(image_dir):
    """Get list of valid input files from directory.

    Args:
        image_dir (str): Directory containing input files

    Returns:
        list: Full paths of the input files, in name order (P5f: not os.listdir order)
    """
    return list_input_files(image_dir)


def determine_file_type(input_files, image_dir):
    """Determine file type and voxel spacing from input files.

    Args:
        input_files (list): List of input file paths
        image_dir (str): Input directory path (unused, kept for callers)

    Returns:
        tuple: (file_type, voxel_spacing)

    Raises:
        CaseInputError: the folder mixes file types. (P5f: this used to pick one of the
        types by hash order, so the same case succeeded or failed depending on the process.)
    """
    file_type = detect_file_type(input_files)
    voxel_spacing = utils_datasets.VOXEL_SPACING if file_type == "png" else None
    return file_type, voxel_spacing


def process_attention_scores(
    prediction,
    serie,
    input_files,
    return_type: str = cfg["RANKING"]["DEFAULT_RETURN_TYPE"],
    top_k: int = cfg["RANKING"]["DEFAULT_TOP_K"],
) -> Dict:
    """Process and rank attention scores with correct index reversal.

    Args:
        prediction: Model prediction object
        serie: Serie object containing images
        input_files (list): List of input file paths
        return_type (str): Type of return:
            - 'all': Return all images (default)
            - 'top': Return top K images
            - 'none': Don't return any images
        top_k (int): Number of top images to return when return_type='top'

    Returns:
        dict: Processed attention information containing:
            - attention_scores: List of dicts with file info and scores
            - total_images: Total number of images processed
            - returned_images: Number of images returned
    """
    # Get ranked images based on attention
    ranked_images = rank_images_by_attention(
        prediction.attentions[0],
        serie.get_raw_images(),
        len(serie.get_raw_images()),
        return_type=return_type,
        top_k=top_k,
    )

    # Return minimal info if return_type is 'none'
    if return_type.lower() == "none" or ranked_images is None:
        return {
            "attention_scores": [],
            "total_images": len(serie.get_raw_images()),
            "returned_images": 0,
        }

    N = len(serie.get_raw_images())
    num_digits = len(str(N))

    # Process attention scores with reversed indexing to match save_attention_images
    attention_scores = []
    for item in ranked_images:
        original_idx = item["original_index"]
        score = item["attention_score"]

        if score > 0:  # Only add images with attention score > 0
            # Calculate the reversed index as used in save_attention_images
            reversed_idx = (N - 1) - original_idx
            # Ensure the index is valid for input_files
            if original_idx < len(input_files):
                # Get original file name and info
                original_file = input_files[original_idx]
                original_filename = os.path.basename(original_file)

                # Extract patient name
                base_name = os.path.splitext(original_filename)[0]
                parts = base_name.split("_")
                file_type = "dcm" if MODEL_CONFIG["SAVE_AS_DICOM_DEFAULT"] else "png"
                if parts and parts[-1].isdigit():
                    patient_name = "_".join(parts[:-1])
                    # Create prediction filename using the reversed index
                    pred_filename = f"{vsl_config['FILE_NAMING']['PREDICTION_PREFIX']}{patient_name}_{reversed_idx:0{num_digits}d}.{file_type}"  # Use .png or .dcm as needed
                else:
                    patient_name = base_name
                    pred_filename = f"{vsl_config['FILE_NAMING']['PREDICTION_PREFIX']}{patient_name}_{reversed_idx:0{num_digits}d}.{file_type}"  # Use .png or .dcm as needed

                attention_scores.append(
                    {
                        "file_name_pred": pred_filename,
                        "attention_score": score,
                    }
                )
            else:
                print(
                    f"⚠️ Warning: Index {original_idx} is out of range for input_files (length {len(input_files)})"
                )

    # Create result with additional information
    result = {
        "attention_scores": attention_scores,
        "total_images": N,
        "returned_images": len(attention_scores),
    }

    # Add information about top_k if used
    if return_type.lower() == "top" and top_k is not None:
        result["top_k_requested"] = top_k

    return result


def predict(
    image_dir,
    output_dir,
    model=None,
    file_type: Literal["auto", "dicom", "png"] = "auto",
    threads: int = 0,
    return_attentions: bool = MODEL_CONFIG["RETURN_ATTENTIONS_DEFAULT"],
    write_attention_images: bool = MODEL_CONFIG["WRITE_ATTENTION_IMAGES_DEFAULT"],
):
    """Run the model prediction.

    Args:
        image_dir (str): The directory of the images to predict.
        output_dir (str): The directory to save the prediction results.
        model (Sybil): The model to use for prediction.
        return_attentions (bool): Whether to return the attention scores.
        write_attention_images (bool): Whether to visualize attention maps
        save_as_dicom (bool): Whether to save visualizations as DICOM
        file_type (str): Type of input files ("auto", "dicom", or "png")
        threads (int): Number of threads to use

    Returns:
        tuple: (prediction_dict, series_with_attention, attention_info)
    """
    logger = logging_utils.get_logger()

    return_attentions |= write_attention_images

    # Get input files
    input_files = get_input_files(image_dir)

    # Determine file type
    file_type, voxel_spacing = determine_file_type(input_files, image_dir)

    logger.debug(f"Processing {len(input_files)} {file_type} files from {image_dir}")

    assert file_type in {"dicom", "png"}
    file_type = typing.cast(typing.Literal["dicom", "png"], file_type)

    # Load model if needed
    if model is None:
        model = load_model()

    # Create Serie and get predictions
    serie = Serie(input_files, voxel_spacing=voxel_spacing, file_type=file_type)
    prediction = model.predict(
        [serie], return_attentions=return_attentions, threads=threads
    )
    prediction_scores = prediction.scores[0]

    logger.debug(f"Prediction finished. Results:\n{prediction_scores}")

    # Save predictions
    prediction_path = os.path.join(output_dir, "prediction_scores.json")
    pred_dict = {"predictions": prediction.scores}
    with open(prediction_path, "w") as f:
        json.dump(pred_dict, f, indent=2)

    series_with_attention = None
    attention_info = None

    # Handle DICOM metadata
    dicom_metadata_list = []
    if file_type == "dicom":
        # First, load all DICOM metadata
        dicom_metadata_dict = {
            os.path.normpath(f): pydicom.dcmread(f) for f in input_files
        }

        # Then, reorder according to the serie's ordered paths
        # This ensures dicom_metadata_list matches the order of images in serie
        ordered_paths = [os.path.normpath(path) for path in serie._meta.paths]
        dicom_metadata_list = [dicom_metadata_dict[path] for path in ordered_paths]

        if not dicom_metadata_list:
            logging.warning("⚠️ No DICOM metadata could be loaded from input files")

    # Process attention scores if requested
    if return_attentions:
        attention_path = os.path.join(output_dir, "attention_scores.pkl")
        with open(attention_path, "wb") as f:
            pickle.dump(prediction, f)

        # P5f: names follow the slice order the model used (serie order), not the listing.
        attention_info = process_attention_scores(prediction, serie, list(serie._meta.paths))

        # Save rankings
        ranking_path = os.path.join(output_dir, "image_ranking.json")
        with open(ranking_path, "w") as f:
            json.dump(attention_info, f, indent=2)

    # Visualize attention if requested
    if write_attention_images:
        series_with_attention = visualize_attentions(
            [serie],
            attentions=prediction.attentions,
            save_directory=output_dir,
            dicom_metadata_list=dicom_metadata_list,
            input_files=list(serie._meta.paths),
            save_as_dicom=MODEL_CONFIG["SAVE_AS_DICOM_DEFAULT"],
            save_original=MODEL_CONFIG["SAVE_ORIGINAL_DEFAULT"],
        )

    return pred_dict, series_with_attention, attention_info
