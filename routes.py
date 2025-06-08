import os
import shutil
import uuid

from flask import Blueprint, jsonify, request, send_file, send_from_directory

from call_model import load_model, predict
from config import FOLDERS, IS_DEV
from utils import (
    cleanup_old_results,
    create_zip_result,
    dicom_to_png,
    extract_zip_file,
    get_file_path,
    get_overlay_files,
    get_valid_files,
    save_uploaded_files,
    save_uploaded_zip,
)

bp = Blueprint("routes", __name__)

model = load_model()


@bp.route("/api_predict", methods=["POST"])
def api_predict():
    """
    API that receives session_id, accesses the pre-extracted folder, runs the model and returns results

    Args:
        session_id (str): Session ID pointing to pre-extracted folder

    Returns:
        JSON: Prediction results including attention information
    """
    data = request.get_json()
    session_id = data.get("session_id") if data else None
    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400

    unzip_path = os.path.join(FOLDERS["UPLOAD"], session_id)
    if not os.path.exists(unzip_path):
        return jsonify({"error": f"Session folder not found: {unzip_path}"}), 404

    valid_files = get_valid_files(unzip_path)
    if not valid_files:
        return jsonify({"error": "No valid files found in the session folder"}), 400

    output_dir = os.path.join(FOLDERS["RESULTS"], session_id, "sybil")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Session ID: {session_id}, Output directory: {output_dir}")

    # Run prediction
    pred_dict, _, attention_info = predict(unzip_path, output_dir, model)

    # DO NOT delete directory after creating zip (keep for other requests)
    response = {
        "session_id": session_id,
        "predictions": pred_dict["predictions"],
        "attention_info": attention_info,
        "message": "Prediction successful.",
    }
    if IS_DEV:
        print(f"Response: {response}")
    return jsonify(response)


@bp.route("/api_predict_file", methods=["POST"])
def api_predict_file():
    """API to receive photos, run the model, and return the prediction

    Args:
        file (FileStorage): The photos to be uploaded

    Returns:
        JSON: The prediction results including the path and attention values
    """
    print("API predict called")

    cleanup_old_results([FOLDERS["CLEANUP"]])

    files = request.files.getlist("file")

    if not files or all(file.filename == "" for file in files):
        return jsonify({"error": "No selected files"}), 400

    # Create a UUID for each prediction request
    session_id = str(uuid.uuid4())

    # Save the files & get the list of uploaded files
    uploaded_files, upload_path = save_uploaded_files(
        files, session_id, folder_save=FOLDERS["CLEANUP"]
    )
    if not uploaded_files:
        return jsonify({"error": "No valid files uploaded"}), 400

    output_dir = os.path.join(FOLDERS["CLEANUP"], session_id)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Session ID: {session_id}, Output directory: {output_dir}")

    # Run prediction
    pred_dict, overlayed_images, attention_info = predict(
        upload_path,
        output_dir,
        model,
    )

    # Get the list of overlay files
    overlay_files = get_overlay_files(output_dir, session_id)
    base_url = request.host_url.rstrip("/")
    overlay_image_info = [
        {
            "filename": img,
            # "download_link": f"{base_url}/download/{session_id}/{img}",
            # "preview_link": f"{base_url}/preview/{session_id}/{img}",
        }
        for img in overlay_files
    ]

    # Return the JSON result including the path and attention values
    response = {
        "session_id": session_id,
        "predictions": pred_dict["predictions"],
        "overlay_images": overlay_image_info,
        "attention_info": attention_info,
        "gif_download": (
            f"{base_url}/download_gif/{session_id}" if overlay_files else None
        ),
        "message": "Prediction successful.",
    }

    return jsonify(response)


@bp.route("/api_predict_zip", methods=["POST"])
def api_predict_zip():
    """
    API that receives ZIP file, unzips it, runs the model and returns results.
    This version also cleans up old results before processing.

    Args:
        file (FileStorage): The ZIP file to be uploaded

    Returns:
        JSON: Prediction results including the ZIP download link and attention information
    """
    # Clean up old results before processing
    cleanup_old_results([FOLDERS["CLEANUP"]])

    file = request.files.get("file")
    if not file or file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not file.filename.endswith(".zip"):
        return jsonify({"error": "Invalid file format. Only ZIP is allowed."}), 400

    print("File upload:", file)
    session_id = str(uuid.uuid4())

    # Save the uploaded ZIP file to CLEANUP_FOLDER
    zip_path = save_uploaded_zip(file, session_id, folder_save=FOLDERS["CLEANUP"])

    # Unzip the ZIP file to CLEANUP_FOLDER
    unzip_path, error_response, status_code = extract_zip_file(
        zip_path, session_id, folder_save=FOLDERS["CLEANUP"]
    )
    if error_response:
        return error_response, status_code

    # Check if there are valid files
    valid_files = get_valid_files(unzip_path)
    if not valid_files:
        shutil.rmtree(unzip_path)  # Delete the empty directory
        return jsonify({"error": "No valid files found in the ZIP archive"}), 400

    # The directory to save the prediction results
    output_dir = os.path.join(FOLDERS["CLEANUP"], session_id)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Session ID: {session_id}, Output directory: {output_dir}")

    # Run prediction
    pred_dict, _, attention_info = predict(unzip_path, output_dir, model)

    # Path to overlay images directory
    overlay_images_link = output_dir  # Changed from os.path.join(output_dir, "serie_0")

    # Check if overlay directory exists and contains images
    if not os.path.exists(overlay_images_link):
        return jsonify({"error": "Overlay images folder not found"}), 500

    overlay_files = [
        f for f in os.listdir(overlay_images_link) if f.endswith(".dcm")
    ]  # Only look for DICOM files
    if not overlay_files:
        return jsonify({"error": "No overlay images generated"}), 500

    print(f"Found {len(overlay_files)} overlay images in {overlay_images_link}")

    # Zip the prediction results
    try:
        zip_path = create_zip_result(
            overlay_images_link, session_id, folder_save=FOLDERS["CLEANUP"]
        )
        print(f"Created zip file at: {zip_path}")

        if not os.path.exists(zip_path) or os.path.getsize(zip_path) == 0:
            return jsonify({"error": "Failed to create zip file"}), 500

    except Exception as e:
        print(f"Error creating zip file: {str(e)}")
        return jsonify({"error": f"Failed to create zip file: {str(e)}"}), 500

    base_url = request.host_url.rstrip("/")
    zip_download_link = f"{base_url}/download_zip/{session_id}"

    # Delete the intermediate directories after creating zip file
    shutil.rmtree(unzip_path)
    shutil.rmtree(output_dir)

    # Return the JSON result including the ZIP download link and attention information
    response = {
        "session_id": session_id,
        "predictions": pred_dict["predictions"],
        "overlay_images": zip_download_link,
        "attention_info": attention_info,
        "message": "Prediction successful.",
    }
    if IS_DEV:
        print(f"Response: {response}")
    return jsonify(response)


@bp.route("/convert-list", methods=["POST"])
def convert_dicom_list():
    """
    API to convert a list of DICOM files to PNG format

    Returns:
        JSON: List of converted images in base64 format
    """
    if "files" not in request.files:
        return jsonify({"error": "No files uploaded"}), 400

    files = request.files.getlist("files")

    if not files:
        return jsonify({"error": "Empty file list"}), 400

    result = []
    for file in files:
        try:
            img_base64 = dicom_to_png(file)
            result.append(
                {"filename": f"{file.filename}.png", "image_base64": img_base64}
            )
        except Exception as e:
            return (
                jsonify({"error": f"Error processing file {file.filename}: {str(e)}"}),
                500,
            )

    return jsonify({"images": result})


@bp.route("/download/<session_id>/<filename>", methods=["GET"])
def download_file(session_id, filename):
    """API to download Overlay image according to Session ID."""
    file_path = get_file_path(session_id, filename)

    if os.path.exists(file_path):
        print(f"✅ File found: {file_path}, preparing download...")
        return send_file(file_path, as_attachment=True)

    print(f"⚠️ File not found: {file_path}")
    return (
        jsonify(
            {"error": "File not found", "session_id": session_id, "filename": filename}
        ),
        404,
    )


@bp.route("/download_zip/<session_id>", methods=["GET"])
def download_zip(session_id):
    """API to download Overlay image according to Session ID."""
    file_path = os.path.join(FOLDERS["CLEANUP"], session_id + ".zip")
    if os.path.exists(file_path):
        print(f"✅ File found: {file_path}, preparing download...")
        return send_file(file_path, as_attachment=True)

    print(f"⚠️ File not found: {file_path}")
    return (
        jsonify({"error": "File not found", "session_id": session_id}),
        404,
    )


@bp.route("/preview/<session_id>/<filename>", methods=["GET"])
def preview_file(session_id, filename):
    """API to preview overlay photos"""
    overlay_dir = os.path.join(FOLDERS["RESULTS"], session_id)
    # PREDICTION_CONFIG["OVERLAY_PATH"]
    file_path = os.path.join(overlay_dir, filename)

    if os.path.exists(file_path):
        print(f"✅ Previewing file: {file_path}")
        return send_from_directory(overlay_dir, filename)

    print(f"⚠️ Preview file not found: {file_path}")
    return (
        jsonify(
            {"error": "File not found", "session_id": session_id, "filename": filename}
        ),
        404,
    )


@bp.route("/download_gif/<session_id>", methods=["GET"])
def download_gif(session_id):
    """API to download the GIF file of Overlay."""
    gif_filename = "serie_0.gif"  # f"{PREDICTION_CONFIG['OVERLAY_PATH']}.gif"

    gif_path = get_file_path(session_id, gif_filename)

    if os.path.exists(gif_path):
        print(f"✅ GIF found: {gif_path}, preparing download...")
        return send_file(gif_path, as_attachment=True)

    print(f"⚠️ GIF not found: {gif_path}")
    return (
        jsonify(
            {
                "error": "GIF file not found",
                "session_id": session_id,
                "filename": gif_filename,
            }
        ),
        404,
    )
