"""Deterministic reading of a case folder (no torch: unit-tested in CI).

The model used to depend on the order `os.listdir` happened to return:
- output file names mixed that order with the sorted slice order,
- slices at the same z position were ordered by an unstable sort (numpy quicksort), which
  can change the prediction,
- with two file types in a folder, the one checked was picked by hash order, so the same
  case succeeded or failed depending on the process.
Here files are listed in name order and slices are sorted on a total key, so a case gives
the same result on every run; a folder mixing file types (which always failed) is refused
with a clear message.
"""
import os
from typing import List, NamedTuple, Optional


class CaseInputError(ValueError):
    """The case folder cannot be read as one CT series (answered 400 by the API)."""


class SliceKey(NamedTuple):
    path: str
    z: float
    instance_number: Optional[int]
    series_uid: Optional[str]


def list_input_files(image_dir: str) -> List[str]:
    """Files of the case folder in name order, hidden files skipped."""
    names = sorted(
        name
        for name in os.listdir(image_dir)
        if not name.startswith(".") and os.path.isfile(os.path.join(image_dir, name))
    )
    if not names:
        raise CaseInputError("No valid files found in the case folder.")
    return [os.path.join(image_dir, name) for name in names]


def detect_file_type(input_files: List[str]) -> str:
    """'dicom' or 'png'. Files without the .png extension are read as DICOM (as before)."""
    kinds = sorted(
        {"png" if os.path.splitext(f)[1].lower() == ".png" else "dicom" for f in input_files}
    )
    if not kinds:
        raise CaseInputError("No valid files found in the case folder.")
    if len(kinds) > 1:
        raise CaseInputError(
            f"The case mixes file types ({', '.join(kinds)}). Upload one DICOM series."
        )
    return kinds[0]


def order_dicom_slices(keys: List[SliceKey]) -> List[SliceKey]:
    """Slices in ascending z; ties broken by InstanceNumber (missing last), series, file name.

    Nothing is refused here: a folder holding several series (the upload page sends a whole
    study folder) was always scored on all its slices together, and still is — only the order
    is now the same on every run. Choosing one series is a product decision (see P5f notes).
    """
    return sorted(
        keys,
        key=lambda k: (
            k.z,
            k.instance_number if k.instance_number is not None else float("inf"),
            k.series_uid or "",
            k.path,
        ),
    )


def series_count(keys: List[SliceKey]) -> int:
    """Number of distinct SeriesInstanceUIDs (slices without one are not counted)."""
    return len({k.series_uid for k in keys if k.series_uid})


def _int_or_none(value) -> Optional[int]:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None  # "1.5", "abc", multi-valued: the number only breaks ties, never fail on it


def slice_key(path: str, header) -> SliceKey:
    """The ordering key of a slice from its (pixel-less) pydicom header."""
    try:
        instance = header.get("InstanceNumber")
    except (TypeError, ValueError):
        instance = None
    series = header.get("SeriesInstanceUID")
    return SliceKey(
        path=path,
        z=float(header.ImagePositionPatient[-1]),
        instance_number=_int_or_none(instance),
        series_uid=str(series) if series else None,
    )
