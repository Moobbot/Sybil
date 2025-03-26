import os
import pickle
import typing
import urllib
import zipfile
import json
from flask import logging
import numpy as np

import pydicom
from config import CHECKPOINT_DIR, CHECKPOINT_URL, MODEL_PATHS, CALIBRATOR_PATH
from sybil.datasets import utils as utils_datasets
from sybil.model import Sybil
from sybil.serie import Serie
from sybil.utils import logging_utils
from typing import Literal
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

def process_attention_scores(prediction, serie, input_files):
    """Process and rank attention scores.
    
    Args:
        prediction: Model prediction object
        serie: Serie object containing images
        input_files (list): List of input file paths
        
    Returns:
        dict: Processed attention information
    """
    ranked_images = rank_images_by_attention(
        prediction.attentions[0],
        serie.get_raw_images(),
        len(serie.get_raw_images()),
    )

    N = len(ranked_images)
    num_digits = len(str(N))

    attention_info = {
        "attention_scores": [
            {
                "file_name_original": os.path.basename(input_files[i]),
                "file_name_pred": (
                    f"pred_{get_patient_name(input_files[i])[0]}_{int(get_patient_name(input_files[i])[1]):0{num_digits}d}.dcm"
                    if get_patient_name(input_files[i])[1]
                    else f"pred_{get_patient_name(input_files[i])[0]}_{(N-1)-i:0{num_digits}d}.dcm"
                ),
                "rank": item["rank"],
                "attention_score": item["attention_score"],
            }
            for i, item in enumerate(ranked_images)
            if item["attention_score"] > 0
        ]
    }
    
    return attention_info

def predict(
    image_dir,
    output_dir,
    model=None,
    return_attentions=True,
    visualize_attentions_img=False,
    save_as_dicom=False,
    file_type: Literal["auto", "dicom", "png"] = "auto",
    threads: int = 0,
):
    """Run the model prediction.

    Args:
        image_dir (str): The directory of the images to predict.
        output_dir (str): The directory to save the prediction results.
        model (Sybil): The model to use for prediction.
        return_attentions (bool): Whether to return the attention scores.
        visualize_attentions_img (bool): Whether to visualize attention maps
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
    prediction_path = os.path.join(output_dir, "prediction_scores.json")
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
        attention_path = os.path.join(output_dir, "attention_scores.pkl")
        with open(attention_path, "wb") as f:
            pickle.dump(prediction, f)

        attention_info = process_attention_scores(prediction, serie, input_files)
        
        # Save rankings
        ranking_path = os.path.join(output_dir, "image_ranking.json")
        with open(ranking_path, "w") as f:
            json.dump(attention_info, f, indent=2)

    # Visualize attention if requested
    if visualize_attentions_img:
        series_with_attention = visualize_attentions(
            [serie],
            attentions=prediction.attentions,
            save_directory=output_dir,
            gain=3,
            save_as_dicom=save_as_dicom,
            dicom_metadata_list=dicom_metadata_list,
            input_files=input_files,
            save_original=True
        )

    return pred_dict, series_with_attention, attention_info
