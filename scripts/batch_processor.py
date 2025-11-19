"""
Helper utilities for batch processing Sybil predictions across subfolders.
"""

from __future__ import annotations

import csv
import os
import zipfile
from datetime import datetime
from typing import Dict, List, Tuple

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


def _write_image_ranking_csv(csv_path: str, attention_scores: List[Dict]) -> None:
    fieldnames = ["rank", "file_name_pred", "attention_score"]
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for idx, item in enumerate(attention_scores, 1):
            writer.writerow(
                {
                    "rank": idx,
                    "file_name_pred": item.get("file_name_pred", ""),
                    "attention_score": item.get("attention_score", ""),
                }
            )


def _extract_prediction_vector(predictions) -> List[float]:
    if not predictions:
        return []
    first = predictions[0]
    if isinstance(first, dict):
        if "scores" in first:
            return list(first["scores"])
        if "values" in first:
            return list(first["values"])
        return []
    if isinstance(first, (list, tuple)):
        return list(first)
    return []


def _write_prediction_summary_csv(
    batch_result_dir: str, rows: List[Tuple[str, List[float]]]
) -> str | None:
    if not rows:
        return None
    max_years = max((len(vector) for _, vector in rows), default=0)
    fieldnames = ["subfolder_name"] + [
        f"year_{idx}" for idx in range(1, max_years + 1)
    ]
    summary_path = os.path.join(batch_result_dir, "prediction_scores.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for subfolder_name, vector in rows:
            row = {"subfolder_name": subfolder_name}
            for idx in range(max_years):
                key = f"year_{idx + 1}"
                row[key] = vector[idx] if idx < len(vector) else ""
            writer.writerow(row)
    return summary_path


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
    prediction_rows: List[Tuple[str, List[float]]] = []

    for idx, subfolder_path in enumerate(subfolders, 1):
        subfolder_name = os.path.basename(subfolder_path)
        print(f"Processing subfolder {idx}/{len(subfolders)}: {subfolder_name}")
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

        try:
            pred_dict, _, attention_info = predict(
                subfolder_path,
                subfolder_result_dir,
                model,
                write_attention_images=True,
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
            continue

        overall_score = _determine_overall_score(pred_dict.get("predictions", []))
        prediction_vector = _extract_prediction_vector(pred_dict.get("predictions", []))
        prediction_rows.append((subfolder_name, prediction_vector))

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

        ranking_csv = os.path.join(
            batch_result_dir, f"{subfolder_name}_image_ranking.csv"
        )
        _write_image_ranking_csv(
            ranking_csv, attention_info.get("attention_scores", [])
            if attention_info
            else []
        )
        csv_files.append(ranking_csv)

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
        prediction_summary = _write_prediction_summary_csv(
            batch_result_dir, prediction_rows
        )
        if prediction_summary:
            zipf.write(prediction_summary, os.path.basename(prediction_summary))
            csv_files.append(prediction_summary)

    return zip_path

