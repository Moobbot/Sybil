import os
import pickle
import typing
import urllib
import zipfile
import json
from flask import logging
import numpy as np
from typing import Literal, Dict

import pydicom
from config import CHECKPOINT_DIR, CHECKPOINT_URL, MODEL_CONFIG, MODEL_PATHS, CALIBRATOR_PATH, PREDICTION_CONFIG, VISUALIZATION_CONFIG
from sybil.datasets import utils as utils_datasets
from sybil.model import Sybil
from sybil.serie import Serie
from sybil.utils import logging_utils
from sybil.utils.visualization import visualize_attentions, rank_images_by_attention


def download_checkpoints():
    """Download and extract checkpoint if not exist."""
    if not os.path.exists(CHECKPOINT_DIR) or not all(
        os.path.exists(p) for p in MODEL_PATHS
    ):
        print(f"Downloading checkpoints from {CHECKPOINT_URL}...")
        zip_path = os.path.join(CHECKPOINT_DIR, "sybil_checkpoints.zip")
        urllib.request.urlretrieve(CHECKPOINT_URL, zip_path)

        print("Extracting checkpoints...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(CHECKPOINT_DIR)
        os.remove(zip_path)
        print("Checkpoints downloaded and extracted successfully.")


def load_model(model_name="sybil_ensemble"):
    """
    Load a trained Sybil model from a checkpoint or a directory of checkpoints.
    If there is no model or calibrator, download from CHECKPOINT_URL.

    Returns:
        Sybil: Model object has loaded.
    """
    # Kiểm tra và tải checkpoints nếu cần
    if not all(os.path.exists(p) for p in MODEL_PATHS) or not os.path.exists(
        CALIBRATOR_PATH
    ):
        print("Model and Calibrator checkpoints not found. Downloading...")
        download_checkpoints()

    print("Loading Sybil model...")
    try:
        model = Sybil(name_or_path=MODEL_PATHS, calibrator_path=CALIBRATOR_PATH)
    except:
        model = Sybil(model_name)
    print("Model loaded successfully.")
    return model


def get_input_files(image_dir):
    """Get list of valid input files from directory.
    
    Args:
        image_dir (str): Directory containing input files
        
    Returns:
        list: List of full paths to input files
    """
    input_files = [
        os.path.join(image_dir, x)
        for x in os.listdir(image_dir)
        if os.path.isfile(os.path.join(image_dir, x))
    ]
    
    if not input_files:
        raise ValueError("⚠️ No valid files found in the directory.")
        
    return input_files

def determine_file_type(input_files, image_dir):
    """Determine file type and voxel spacing from input files.
    
    Args:
        input_files (list): List of input file paths
        image_dir (str): Input directory path
        
    Returns:
        tuple: (file_type, voxel_spacing)
    """
    voxel_spacing = None
    file_type = "auto"
    
    if file_type == "auto":
        extensions = {os.path.splitext(x)[1] for x in input_files}
        if not extensions:
            raise ValueError("⚠️ No files with valid extensions found.")
        extension = extensions.pop()
        if len(extensions) > 1:
            raise ValueError(
                f"⚠️ Multiple file types found in {image_dir}: {','.join(extensions)}"
            )

        file_type = "dicom" if extension.lower() not in {".png"} else "png"
        if file_type == "png":
            voxel_spacing = utils_datasets.VOXEL_SPACING
            
    return file_type, voxel_spacing

def get_patient_name(file_name):
    """Extract patient name and number from filename.
    
    Args:
        file_name (str): Input filename
        
    Returns:
        tuple: (base_name, number)
    """
    base_name = os.path.splitext(os.path.basename(file_name))[0]
    parts = base_name.split("_")
    if parts and parts[-1].isdigit():
        return "_".join(parts[:-1]), parts[-1]
    else:
        return base_name, ""

def process_attention_scores(
    prediction, 
    serie, 
    input_files,
    return_type: str = VISUALIZATION_CONFIG["RANKING"]["DEFAULT_RETURN_TYPE"],
    top_k: int = VISUALIZATION_CONFIG["RANKING"]["DEFAULT_TOP_K"]
) -> Dict:
    """Process and rank attention scores.
    
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
        top_k=top_k
    )

    # Return minimal info if return_type is 'none'
    if return_type.lower() == 'none' or ranked_images is None:
        return {
            "attention_scores": [],
            "total_images": len(serie.get_raw_images()),
            "returned_images": 0
        }

    N = len(serie.get_raw_images())
    num_digits = len(str(N))

    # Process attention scores
    attention_scores = []
    for item in ranked_images:
        idx = item["original_index"]
        score = item["attention_score"]
        
        if score > 0:  # Chỉ thêm ảnh có attention score > 0
            patient_name_info = get_patient_name(input_files[idx])
            
            # Tạo tên file dự đoán
            if patient_name_info[1]:  # Nếu có số thứ tự
                pred_filename = f"pred_{patient_name_info[0]}_{int(patient_name_info[1]):0{num_digits}d}.dcm"
            else:
                pred_filename = f"pred_{patient_name_info[0]}_{(N-1)-idx:0{num_digits}d}.dcm"
            
            attention_scores.append({
                "file_name_original": os.path.basename(input_files[idx]),
                "file_name_pred": pred_filename,
                "rank": item["rank"],
                "attention_score": score,
            })

    # Tạo kết quả trả về với thông tin bổ sung
    result = {
        "attention_scores": attention_scores,
        "total_images": N,
        "returned_images": len(attention_scores)
    }

    # Thêm thông tin về top_k nếu được sử dụng
    if return_type.lower() == 'top' and top_k is not None:
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
    prediction = model.predict([serie], return_attentions=return_attentions, threads=threads)
    prediction_scores = prediction.scores[0]

    logger.debug(f"Prediction finished. Results:\n{prediction_scores}")

    # Save predictions
    prediction_path = PREDICTION_CONFIG["PREDICTION_PATH"]
    pred_dict = {"predictions": prediction.scores}
    with open(prediction_path, "w") as f:
        json.dump(pred_dict, f, indent=2)

    series_with_attention = None
    attention_info = None

    # Handle DICOM metadata
    dicom_metadata_list = []
    if file_type == "dicom":
        dicom_metadata_list = [pydicom.dcmread(f) for f in input_files]
        if not dicom_metadata_list:
            logging.warning("⚠️ No DICOM metadata could be loaded from input files")

    # Process attention scores if requested
    if return_attentions:
        attention_path = PREDICTION_CONFIG["ATTENTION_PATH"]
        with open(attention_path, "wb") as f:
            pickle.dump(prediction, f)

        attention_info = process_attention_scores(prediction, serie, input_files)
        
        # Save rankings
        ranking_path = PREDICTION_CONFIG["RANKING_PATH"]
        with open(ranking_path, "w") as f:
            json.dump(attention_info, f, indent=2)

    # Visualize attention if requested
    if write_attention_images:
        series_with_attention = visualize_attentions(
            [serie],
            attentions=prediction.attentions,
            save_directory=output_dir,
            dicom_metadata_list=dicom_metadata_list,
            input_files=input_files,
            save_as_dicom=MODEL_CONFIG["SAVE_AS_DICOM_DEFAULT"],
            save_original=MODEL_CONFIG["SAVE_ORIGINAL_DEFAULT"]
        )

    return pred_dict, series_with_attention, attention_info