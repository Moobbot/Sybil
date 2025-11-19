import os
import shutil
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from scripts.batch_processor import BatchProcessError, run_batch_process
from scripts.check_dicom_metadata import check_dicom_folder

from call_model import predict
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

router = APIRouter()
model: Optional[object] = None


def _ensure_model_loaded():
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")


class SessionRequest(BaseModel):
    session_id: str


class BatchProcessRequest(BaseModel):
    folder_path: str


class MetadataCheckRequest(BaseModel):
    folder_path: str
    sample_limit: int = 20


@router.post("/api_predict")
def api_predict(payload: SessionRequest):
    """Run prediction for an existing session folder."""
    _ensure_model_loaded()
    session_id = payload.session_id

    unzip_path = os.path.join(FOLDERS["UPLOAD"], session_id)
    if not os.path.exists(unzip_path):
        raise HTTPException(
            status_code=404, detail=f"Session folder not found: {unzip_path}"
        )

    valid_files = get_valid_files(unzip_path)
    if not valid_files:
        raise HTTPException(
            status_code=400,
            detail="No valid files found in the session folder",
        )

    output_dir = os.path.join(FOLDERS["RESULTS"], session_id, "sybil")
    os.makedirs(output_dir, exist_ok=True)

    pred_dict, _, attention_info = predict(unzip_path, output_dir, model)

    response = {
        "session_id": session_id,
        "predictions": pred_dict["predictions"],
        "attention_info": attention_info,
        "message": "Prediction successful.",
    }
    if IS_DEV:
        print(f"Response: {response}")
    return response


@router.post("/api_predict_file")
def api_predict_file(request: Request, files: List[UploadFile] = File(...)):
    """Upload individual DICOM/PNG files for prediction."""
    _ensure_model_loaded()
    cleanup_old_results([FOLDERS["CLEANUP"]])

    if not files or all(not file.filename for file in files):
        raise HTTPException(status_code=400, detail="No selected files")

    session_id = str(uuid.uuid4())
    uploaded_files, upload_path = save_uploaded_files(
        files, session_id, folder_save=FOLDERS["CLEANUP"]
    )
    if not uploaded_files:
        raise HTTPException(status_code=400, detail="No valid files uploaded")

    output_dir = os.path.join(FOLDERS["CLEANUP"], session_id)
    os.makedirs(output_dir, exist_ok=True)

    pred_dict, _, attention_info = predict(upload_path, output_dir, model)

    overlay_files = get_overlay_files(output_dir, session_id)
    base_url = str(request.base_url).rstrip("/")
    overlay_image_info = [
        {
            "filename": img,
        }
        for img in overlay_files
    ]

    return {
        "session_id": session_id,
        "predictions": pred_dict["predictions"],
        "overlay_images": overlay_image_info,
        "attention_info": attention_info,
        "gif_download": (
            f"{base_url}/download_gif/{session_id}" if overlay_files else None
        ),
        "message": "Prediction successful.",
    }


@router.post("/api_predict_zip")
def api_predict_zip(request: Request, file: UploadFile = File(...)):
    """Upload a ZIP archive containing DICOM images and return prediction results."""
    _ensure_model_loaded()
    cleanup_old_results([FOLDERS["CLEANUP"]])

    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No selected file")

    if not file.filename.endswith(".zip"):
        raise HTTPException(
            status_code=400, detail="Invalid file format. Only ZIP is allowed."
        )

    session_id = str(uuid.uuid4())
    zip_path = save_uploaded_zip(file, session_id, folder_save=FOLDERS["CLEANUP"])

    try:
        unzip_path = extract_zip_file(
            zip_path, session_id, folder_save=FOLDERS["CLEANUP"]
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    valid_files = get_valid_files(unzip_path)
    if not valid_files:
        shutil.rmtree(unzip_path)
        raise HTTPException(
            status_code=400, detail="No valid files found in the ZIP archive"
        )

    output_dir = os.path.join(FOLDERS["CLEANUP"], session_id)
    os.makedirs(output_dir, exist_ok=True)

    pred_dict, _, attention_info = predict(unzip_path, output_dir, model)

    overlay_images_link = output_dir
    if not os.path.exists(overlay_images_link):
        raise HTTPException(status_code=500, detail="Overlay images folder not found")

    overlay_files = [
        f for f in os.listdir(overlay_images_link) if f.lower().endswith(".dcm")
    ]
    if not overlay_files:
        raise HTTPException(status_code=500, detail="No overlay images generated")

    try:
        zip_path = create_zip_result(
            overlay_images_link, session_id, folder_save=FOLDERS["CLEANUP"]
        )
        if not os.path.exists(zip_path) or os.path.getsize(zip_path) == 0:
            raise HTTPException(status_code=500, detail="Failed to create zip file")
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to create zip file: {exc}"
        ) from exc

    base_url = str(request.base_url).rstrip("/")
    zip_download_link = f"{base_url}/download_zip/{session_id}"

    shutil.rmtree(unzip_path)
    shutil.rmtree(output_dir)

    response = {
        "session_id": session_id,
        "predictions": pred_dict["predictions"],
        "overlay_images": zip_download_link,
        "attention_info": attention_info,
        "message": "Prediction successful.",
    }
    if IS_DEV:
        print(f"Response: {response}")
    return response


@router.post("/convert-list")
def convert_dicom_list(files: List[UploadFile] = File(...)):
    """Convert uploaded DICOM files to base64 PNG images."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    result = []
    for file in files:
        try:
            file.file.seek(0)
            img_base64 = dicom_to_png(file.file)
            result.append(
                {"filename": f"{file.filename}.png", "image_base64": img_base64}
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Error processing file {file.filename}: {exc}",
            ) from exc

    return {"images": result}


@router.get("/download/{session_id}/{filename}")
def download_file(session_id: str, filename: str):
    """Download an overlay image by session and filename."""
    file_path = get_file_path(session_id, filename)
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail={
                "error": "File not found",
                "session_id": session_id,
                "filename": filename,
            },
        )

    return FileResponse(file_path, filename=filename)


@router.get("/download_zip/{session_id}")
def download_zip(session_id: str):
    """Download the generated ZIP for a session."""
    file_path = os.path.join(FOLDERS["CLEANUP"], f"{session_id}.zip")
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail={"error": "File not found", "session_id": session_id},
        )

    return FileResponse(file_path, filename=f"{session_id}_results.zip")


@router.get("/preview/{session_id}/{filename}")
def preview_file(session_id: str, filename: str):
    """Preview overlay image inline."""
    overlay_dir = os.path.join(FOLDERS["RESULTS"], session_id)
    file_path = os.path.join(overlay_dir, filename)
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail={
                "error": "File not found",
                "session_id": session_id,
                "filename": filename,
            },
        )

    return FileResponse(file_path)


@router.get("/download_gif/{session_id}")
def download_gif(session_id: str):
    """Download generated GIF."""
    gif_filename = "serie_0.gif"
    gif_path = get_file_path(session_id, gif_filename)
    if not os.path.exists(gif_path):
        raise HTTPException(
            status_code=404,
            detail={
                "error": "GIF file not found",
                "session_id": session_id,
                "filename": gif_filename,
            },
        )

    return FileResponse(
        gif_path, filename=f"{session_id}_results.gif", media_type="image/gif"
    )


@router.post("/api_batch_process")
def api_batch_process(payload: BatchProcessRequest):
    """Batch process all subfolders inside a directory and return consolidated CSVs."""
    _ensure_model_loaded()
    try:
        zip_path = run_batch_process(payload.folder_path, FOLDERS["CLEANUP"], model)
    except BatchProcessError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Batch processing failed: {exc}"
        ) from exc

    filename = os.path.basename(zip_path)
    return FileResponse(zip_path, filename=filename, media_type="application/zip")


@router.post("/api_check_metadata")
def api_check_metadata(payload: MetadataCheckRequest):
    """Inspect DICOM metadata availability and slice ordering for a given folder."""
    folder_path = payload.folder_path
    if not os.path.exists(folder_path):
        raise HTTPException(status_code=404, detail=f"Folder not found: {folder_path}")
    if not os.path.isdir(folder_path):
        raise HTTPException(
            status_code=400, detail=f"Path is not a directory: {folder_path}"
        )

    summary = check_dicom_folder(
        Path(folder_path), show=max(1, min(payload.sample_limit, 200))
    )
    summary["example_folder"] = "C:/Users/ngota/Downloads/test"

    return JSONResponse(summary)
