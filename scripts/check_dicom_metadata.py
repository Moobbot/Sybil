"""
Utility script to inspect DICOM metadata completeness and slice ordering.

Usage:
    python scripts/check_dicom_metadata.py --folder path/to/dicom
"""

from __future__ import annotations

import argparse
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pydicom


def scan_dicom_files(folder: Path) -> List[Dict]:
    """Read DICOM headers once and capture metadata details per file."""
    records: List[Dict] = []
    for entry in sorted(folder.iterdir()):
        if not entry.is_file() or entry.suffix.lower() != ".dcm":
            continue
        dcm = pydicom.dcmread(str(entry), stop_before_pixels=True)
        ipp = getattr(dcm, "ImagePositionPatient", None)
        slice_location = getattr(dcm, "SliceLocation", None)
        instance_number = getattr(dcm, "InstanceNumber", None)
        records.append(
            {
                "file_name": entry.name,
                "has_image_position": bool(ipp),
                "image_position_z": float(ipp[2]) if ipp else "",
                "slice_location": float(slice_location)
                if slice_location is not None
                else "",
                "instance_number": int(instance_number)
                if instance_number is not None
                else "",
            }
        )
    return records


def check_dicom_folder(folder: Path, show: int = 20) -> dict:
    """Run metadata checks for a folder and return structured summary."""
    records = scan_dicom_files(folder)
    missing = [r["file_name"] for r in records if not r["has_image_position"]]
    positions = [
        (r["file_name"], r["image_position_z"])
        for r in records
        if r["has_image_position"]
    ]
    positions.sort(key=lambda item: item[1])

    summary = {
        "folder": str(folder),
        "total_files": len(records),
        "missing_count": len(missing),
        "missing_samples": missing[:show],
        "position_count": len(positions),
        "positions_first": positions[:show],
        "positions_last": positions[-show:] if positions else [],
        "warnings": [],
    }

    if not positions:
        summary["warnings"].append(
            "No slices contained ImagePositionPatient; order may rely on fallback metadata."
        )
    if missing:
        summary["warnings"].append(
            "Some slices are missing ImagePositionPatient — verify ordering with InstanceNumber/SliceLocation."
        )
    if not summary["warnings"]:
        summary["warnings"].append("All slices contain ImagePositionPatient metadata.")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Inspect DICOM metadata in a folder.")
    parser.add_argument(
        "--folder",
        required=True,
        help="Path to folder containing DICOM files",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=20,
        help="Number of file names to show for diagnostics",
    )
    parser.add_argument(
        "--csv",
        help="Optional path to save per-file metadata report (CSV). Defaults to folder/dicom_metadata_report_<timestamp>.csv",
    )
    args = parser.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"Folder not found: {folder}")

    print(f"Scanning DICOM folder: {folder}")

    records = scan_dicom_files(folder)
    summary = check_dicom_folder(folder, show=args.show)
    summary["report_total_records"] = len(records)

    print(json.dumps(summary, indent=2))

    csv_path = (
        Path(args.csv).expanduser().resolve()
        if args.csv
        else folder / f"dicom_metadata_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    if records:
        fieldnames = [
            "file_name",
            "has_image_position",
            "image_position_z",
            "slice_location",
            "instance_number",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        print(f"Saved detailed report to: {csv_path}")
    else:
        print("No DICOM files found; CSV report not generated.")


if __name__ == "__main__":
    main()

