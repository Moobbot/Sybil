import os
import shutil
import uuid
import csv
import zipfile
from datetime import datetime

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


@bp.route("/api_batch_process", methods=["POST"])
def api_batch_process():
    """
    API xử lý batch: đọc tuần tự từng subfolder trong folder được chọn và thực hiện dự đoán.
    
    Request body:
    {
        "folder_path": "C:/path/to/folder/containing/subfolders"
    }
    
    Returns:
        ZIP file chứa các CSV files (mỗi subfolder 1 CSV với kết quả từng ảnh) 
        và 1 file summary CSV tổng hợp
    """
    print("API batch process called")
    
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing request body"}), 400
    
    folder_path = data.get("folder_path")
    if not folder_path:
        return jsonify({"error": "Missing folder_path"}), 400
    
    # Kiểm tra folder có tồn tại không
    if not os.path.exists(folder_path):
        return jsonify({"error": f"Folder not found: {folder_path}"}), 404
    
    if not os.path.isdir(folder_path):
        return jsonify({"error": f"Path is not a directory: {folder_path}"}), 400
    
    # Kiểm tra model có sẵn không
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    # Lấy danh sách các subfolder
    subfolders = []
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            subfolders.append(item_path)
    
    if not subfolders:
        return jsonify({"error": "No subfolders found in the specified folder"}), 400
    
    print(f"Found {len(subfolders)} subfolders to process")
    
    # Tạo thư mục kết quả cho batch processing
    batch_session_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    batch_result_dir = os.path.join(FOLDERS["CLEANUP"], batch_session_id)
    os.makedirs(batch_result_dir, exist_ok=True)
    
    # Danh sách CSV files đã tạo
    csv_files = []
    summary_results = []
    
    # Xử lý từng subfolder tuần tự
    for idx, subfolder_path in enumerate(subfolders, 1):
        subfolder_name = os.path.basename(subfolder_path)
        print(f"Processing subfolder {idx}/{len(subfolders)}: {subfolder_name}")
        
        try:
            # Kiểm tra file hợp lệ trong subfolder
            valid_files = get_valid_files(subfolder_path)
            
            if not valid_files:
                print(f"Warning: No valid files found in subfolder: {subfolder_name}")
                # Tạo CSV rỗng cho subfolder này
                csv_filename = f"{subfolder_name}_results.csv"
                csv_path = os.path.join(batch_result_dir, csv_filename)
                with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
                    writer = csv.DictWriter(
                        csvfile,
                        fieldnames=["file_name", "attention_score", "status", "error_message"]
                    )
                    writer.writeheader()
                    writer.writerow({
                        "file_name": "",
                        "attention_score": "",
                        "status": "error",
                        "error_message": "No valid DICOM/PNG files found"
                    })
                csv_files.append(csv_path)
                summary_results.append({
                    "subfolder_name": subfolder_name,
                    "status": "error",
                    "error_message": "No valid DICOM/PNG files found",
                    "csv_file": csv_filename
                })
                continue
            
            # Tạo thư mục kết quả cho subfolder này
            subfolder_result_dir = os.path.join(batch_result_dir, subfolder_name)
            os.makedirs(subfolder_result_dir, exist_ok=True)
            
            # Thực hiện dự đoán
            try:
                pred_dict, _, attention_info = predict(
                    subfolder_path,
                    subfolder_result_dir,
                    model
                )
                
                # Lấy điểm số tổng thể (Sybil trả về scores là list)
                prediction_scores = pred_dict.get("predictions", [])
                overall_score = None
                if prediction_scores and len(prediction_scores) > 0:
                    # Sybil có thể trả về nhiều scores, lấy score đầu tiên
                    if isinstance(prediction_scores[0], (list, tuple)):
                        overall_score = prediction_scores[0][0] if len(prediction_scores[0]) > 0 else None
                    else:
                        overall_score = prediction_scores[0]
                
                # Tạo CSV cho subfolder này với thông tin từng ảnh
                csv_filename = f"{subfolder_name}_results.csv"
                csv_path = os.path.join(batch_result_dir, csv_filename)
                
                with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
                    fieldnames = [
                        "file_name",
                        "attention_score",
                        "overall_score",
                        "subfolder_name",
                        "subfolder_path"
                    ]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    # Ghi thông tin từng ảnh
                    attention_scores = attention_info.get("attention_scores", []) if attention_info else []
                    if attention_scores:
                        for img_info in attention_scores:
                            writer.writerow({
                                "file_name": img_info.get("file_name_pred", ""),
                                "attention_score": img_info.get("attention_score", 0),
                                "overall_score": overall_score,
                                "subfolder_name": subfolder_name,
                                "subfolder_path": subfolder_path
                            })
                    else:
                        # Nếu không có attention scores, vẫn tạo CSV với thông tin tổng thể
                        writer.writerow({
                            "file_name": "N/A",
                            "attention_score": "",
                            "overall_score": overall_score,
                            "subfolder_name": subfolder_name,
                            "subfolder_path": subfolder_path
                        })
                
                csv_files.append(csv_path)
                summary_results.append({
                    "subfolder_name": subfolder_name,
                    "status": "success",
                    "overall_score": overall_score,
                    "total_images": attention_info.get("total_images", 0) if attention_info else 0,
                    "returned_images": attention_info.get("returned_images", 0) if attention_info else 0,
                    "csv_file": csv_filename
                })
                
                print(f"Successfully processed subfolder: {subfolder_name}, score: {overall_score}, CSV created: {csv_filename}")
                
            except Exception as e:
                error_msg = str(e)
                print(f"Error processing subfolder {subfolder_name}: {error_msg}")
                import traceback
                traceback.print_exc()
                
                # Tạo CSV với thông báo lỗi
                csv_filename = f"{subfolder_name}_results.csv"
                csv_path = os.path.join(batch_result_dir, csv_filename)
                with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
                    writer = csv.DictWriter(
                        csvfile,
                        fieldnames=["file_name", "attention_score", "status", "error_message"]
                    )
                    writer.writeheader()
                    writer.writerow({
                        "file_name": "",
                        "attention_score": "",
                        "status": "error",
                        "error_message": error_msg
                    })
                csv_files.append(csv_path)
                summary_results.append({
                    "subfolder_name": subfolder_name,
                    "status": "error",
                    "error_message": error_msg,
                    "csv_file": csv_filename
                })
                
        except Exception as e:
            error_msg = str(e)
            print(f"Unexpected error processing subfolder {subfolder_name}: {error_msg}")
            import traceback
            traceback.print_exc()
            
            # Tạo CSV với thông báo lỗi
            csv_filename = f"{subfolder_name}_results.csv"
            csv_path = os.path.join(batch_result_dir, csv_filename)
            try:
                with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
                    writer = csv.DictWriter(
                        csvfile,
                        fieldnames=["file_name", "attention_score", "status", "error_message"]
                    )
                    writer.writeheader()
                    writer.writerow({
                        "file_name": "",
                        "attention_score": "",
                        "status": "error",
                        "error_message": error_msg
                    })
                csv_files.append(csv_path)
                summary_results.append({
                    "subfolder_name": subfolder_name,
                    "status": "error",
                    "error_message": error_msg,
                    "csv_file": csv_filename
                })
            except Exception as csv_error:
                print(f"Error creating error CSV for {subfolder_name}: {csv_error}")
    
    # Tạo file ZIP chứa tất cả các CSV files
    zip_filename = f"{batch_session_id}_results.zip"
    zip_path = os.path.join(batch_result_dir, zip_filename)
    
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for csv_file in csv_files:
                if os.path.exists(csv_file):
                    arcname = os.path.basename(csv_file)
                    zipf.write(csv_file, arcname)
                    print(f"Added {arcname} to ZIP")
        
        # Tạo file summary CSV
        summary_csv_filename = f"{batch_session_id}_summary.csv"
        summary_csv_path = os.path.join(batch_result_dir, summary_csv_filename)
        with open(summary_csv_path, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "subfolder_name",
                "status",
                "overall_score",
                "total_images",
                "returned_images",
                "error_message",
                "csv_file"
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in summary_results:
                writer.writerow(result)
        
        # Thêm summary CSV vào ZIP
        with zipfile.ZipFile(zip_path, "a", zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(summary_csv_path, summary_csv_filename)
        
        print(f"ZIP file created: {zip_path}")
        print(f"Total processed: {len(summary_results)}, Successful: {sum(1 for r in summary_results if r['status'] == 'success')}, Failed: {sum(1 for r in summary_results if r['status'] == 'error')}")
        
        # Trả về file ZIP
        return send_file(zip_path, as_attachment=True, download_name=zip_filename)
        
    except Exception as e:
        print(f"Error creating ZIP file: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Failed to create ZIP file: {str(e)}"}), 500
