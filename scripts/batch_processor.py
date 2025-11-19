"""
Helper utilities for batch processing Sybil predictions across subfolders.
"""

from __future__ import annotations

import csv
import os
import zipfile
from datetime import datetime
from typing import List, Tuple

from call_model import predict
from utils import get_valid_files


class BatchProcessError(ValueError):
    """Raised for user-facing batch processing errors."""


def _determine_overall_score(predictions):
    if not predictions:
        return None
    first = predictions[0]
    if isinstance(first, (list, tuple)):
        return first[0] if first else None
    if isinstance(first, dict):
        return first.get("score")
    return first


def _write_error_csv(csv_path: str, error_message: str) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["file_name", "attention_score", "status", "error_message"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "file_name": "",
                "attention_score": "",
                "status": "error",
                "error_message": error_message,
            }
        )


def run_batch_process(folder_path: str, cleanup_dir: str, model) -> str:
    """
    Execute batch predictions for all subfolders inside folder_path.

    Returns the absolute path to the generated ZIP archive containing CSV outputs.
    """
    if model is None:
        raise BatchProcessError("Model not loaded")

    if not os.path.exists(folder_path):
        raise BatchProcessError(f"Folder not found: {folder_path}")
    if not os.path.isdir(folder_path):
        raise BatchProcessError(f"Path is not a directory: {folder_path}")

    subfolders = [
        os.path.join(folder_path, item)
        for item in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, item))
    ]
    if not subfolders:
        raise BatchProcessError("No subfolders found in the specified folder")

    batch_session_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    batch_result_dir = os.path.join(cleanup_dir, batch_session_id)
    os.makedirs(batch_result_dir, exist_ok=True)

    csv_files: List[str] = []
    summary_results: List[dict] = []

    for idx, subfolder_path in enumerate(subfolders, 1):
        subfolder_name = os.path.basename(subfolder_path)
        print(f"Processing subfolder {idx}/{len(subfolders)}: {subfolder_name}")
        try:
            valid_files = get_valid_files(subfolder_path)
            if not valid_files:
                csv_filename = f"{subfolder_name}_results.csv"
                csv_path = os.path.join(batch_result_dir, csv_filename)
                _write_error_csv(csv_path, "No valid DICOM/PNG files found")
                csv_files.append(csv_path)
                summary_results.append(
                    {
                        "subfolder_name": subfolder_name,
                        "status": "error",
                        "error_message": "No valid DICOM/PNG files found",
                        "csv_file": csv_filename,
                    }
                )
                continue

            subfolder_result_dir = os.path.join(batch_result_dir, subfolder_name)
            os.makedirs(subfolder_result_dir, exist_ok=True)

            pred_dict, _, attention_info = predict(
                subfolder_path,
                subfolder_result_dir,
                model,
            )

            overall_score = _determine_overall_score(pred_dict.get("predictions", []))

            csv_filename = f"{subfolder_name}_results.csv"
            csv_path = os.path.join(batch_result_dir, csv_filename)
            with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
                fieldnames = [
                    "file_name",
                    "attention_score",
                    "overall_score",
                    "subfolder_name",
                    "subfolder_path",
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                attention_scores = (
                    attention_info.get("attention_scores", []) if attention_info else []
                )
                if attention_scores:
                    for img_info in attention_scores:
                        writer.writerow(
                            {
                                "file_name": img_info.get("file_name_pred", ""),
                                "attention_score": img_info.get("attention_score", 0),
                                "overall_score": overall_score,
                                "subfolder_name": subfolder_name,
                                "subfolder_path": subfolder_path,
                            }
                        )
                else:
                    writer.writerow(
                        {
                            "file_name": "N/A",
                            "attention_score": "",
                            "overall_score": overall_score,
                            "subfolder_name": subfolder_name,
                            "subfolder_path": subfolder_path,
                        }
                    )

            csv_files.append(csv_path)
            summary_results.append(
                {
                    "subfolder_name": subfolder_name,
                    "status": "success",
                    "overall_score": overall_score,
                    "total_images": attention_info.get("total_images", 0)
                    if attention_info
                    else 0,
                    "returned_images": attention_info.get("returned_images", 0)
                    if attention_info
                    else 0,
                    "csv_file": csv_filename,
                }
            )

        except Exception as exc:
            csv_filename = f"{subfolder_name}_results.csv"
            csv_path = os.path.join(batch_result_dir, csv_filename)
            _write_error_csv(csv_path, str(exc))
            csv_files.append(csv_path)
            summary_results.append(
                {
                    "subfolder_name": subfolder_name,
                    "status": "error",
                    "error_message": str(exc),
                    "csv_file": csv_filename,
                }
            )

    zip_filename = f"{batch_session_id}_results.zip"
    zip_path = os.path.join(batch_result_dir, zip_filename)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for csv_file in csv_files:
            if os.path.exists(csv_file):
                arcname = os.path.basename(csv_file)
                zipf.write(csv_file, arcname)

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
            "csv_file",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in summary_results:
            writer.writerow(result)

    with zipfile.ZipFile(zip_path, "a", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(summary_csv_path, summary_csv_filename)

    return zip_path

