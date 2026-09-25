"""P5f — deterministic reading of a case folder (input_series.py). No torch needed."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from input_series import (  # noqa: E402
    CaseInputError,
    SliceKey,
    detect_file_type,
    list_input_files,
    order_dicom_slices,
    series_count,
    slice_key,
)


def touch(d, name):
    p = d / name
    p.write_bytes(b"x")
    return str(p)


def test_files_in_name_order_whatever_the_listing_order(tmp_path, monkeypatch):
    for n in ["img_00003.dcm", "img_00001.dcm", "img_00002.dcm", ".DS_Store"]:
        touch(tmp_path, n)
    (tmp_path / "subdir").mkdir()
    listing = ["img_00002.dcm", ".DS_Store", "subdir", "img_00003.dcm", "img_00001.dcm"]
    monkeypatch.setattr(os, "listdir", lambda _d: list(listing))
    got = [os.path.basename(p) for p in list_input_files(str(tmp_path))]
    assert got == ["img_00001.dcm", "img_00002.dcm", "img_00003.dcm"]


def test_empty_folder_is_a_case_error(tmp_path):
    touch(tmp_path, ".hidden")
    with pytest.raises(CaseInputError):
        list_input_files(str(tmp_path))


@pytest.mark.parametrize(
    "names,kind",
    [
        (["a.dcm", "b.dcm"], "dicom"),
        (["a.dcm", "b.DCM", "c"], "dicom"),  # extensionless entries are DICOM, as before
        (["a.png", "b.PNG"], "png"),
    ],
)
def test_file_type(names, kind):
    assert detect_file_type(names) == kind


def test_mixed_file_types_always_refused_and_named():
    # Regression: the type was pop()-ed from a set BEFORE checking, so with two types one was
    # picked by hash order — the same case succeeded or failed depending on the process.
    for order in (["a.png", "b.dcm"], ["b.dcm", "a.png"]):
        with pytest.raises(CaseInputError, match="dicom, png"):
            detect_file_type(order)


def k(path, z, inst=None, series="1.2.3"):
    return SliceKey(path, z, inst, series)


def test_slices_sorted_by_z():
    got = order_dicom_slices([k("c", 3.0), k("a", 1.0), k("b", 2.0)])
    assert [s.path for s in got] == ["a", "b", "c"]


def test_same_z_is_ordered_by_instance_whatever_the_input_order():
    # Regression: numpy argsort (quicksort) on z alone ordered tied slices arbitrarily.
    slices = [k("x", 5.0, 2), k("y", 5.0, 1), k("a", 1.0, 9), k("m", 7.0, None), k("l", 7.0, 4)]
    expected = ["a", "y", "x", "l", "m"]  # a slice without InstanceNumber goes after numbered ones
    for rotation in range(len(slices)):
        rotated = slices[rotation:] + slices[:rotation]
        assert [s.path for s in order_dicom_slices(rotated)] == expected
        assert [s.path for s in order_dicom_slices(list(reversed(rotated)))] == expected


def test_several_series_are_kept_together_as_before_in_a_fixed_order():
    # The upload page sends whole study folders: refusing them would stop cases that are
    # scored today. The order of their tied slices is now fixed (series, then file name).
    slices = [k("b", 1.0, 1, series="2.2"), k("a", 1.0, 1, series="1.1"), k("c", 0.5, 3, series="2.2")]
    for order in (slices, list(reversed(slices))):
        assert [s.path for s in order_dicom_slices(order)] == ["c", "a", "b"]
    assert series_count(slices) == 2


def test_duplicate_files_are_kept_in_a_fixed_order():
    for order in ([k("b", 1.0, 3), k("a", 1.0, 3)], [k("a", 1.0, 3), k("b", 1.0, 3)]):
        assert [s.path for s in order_dicom_slices(order)] == ["a", "b"]


def test_missing_series_uid_does_not_count_as_a_series():
    assert series_count([k("a", 1.0, series=None), k("b", 2.0, series="1.1")]) == 1


def test_slice_key_from_a_real_header(tmp_path):
    pydicom = pytest.importorskip("pydicom")
    from pydicom.dataset import Dataset

    ds = Dataset()
    ds.ImagePositionPatient = [-10.0, 20.0, "-12.5"]
    ds.InstanceNumber = "7"
    ds.SeriesInstanceUID = "1.2.840.1"
    got = slice_key("p", ds)
    assert got == SliceKey("p", -12.5, 7, "1.2.840.1")

    bare = Dataset()
    bare.ImagePositionPatient = [0, 0, 4]
    assert slice_key("q", bare) == SliceKey("q", 4.0, None, None)
    assert pydicom  # imported for the Dataset type


class FakeHeader:
    """A pydicom-like header: .get() may raise, as pydicom does on a malformed IS value."""

    def __init__(self, instance, raises=False):
        self.ImagePositionPatient = [0, 0, 1]
        self._instance, self._raises = instance, raises

    def get(self, name):
        if name == "InstanceNumber":
            if self._raises:
                raise TypeError("Could not convert value to integer")
            return self._instance
        return None


@pytest.mark.parametrize("instance,raises", [("1.5", False), ("abc", False), (["1", "2"], False), (None, True)])
def test_a_malformed_instance_number_is_ignored_not_fatal(instance, raises):
    # Review M1: the number only breaks ties; a case with a broken one used to be scored and must still be.
    assert slice_key("p", FakeHeader(instance, raises)).instance_number is None


@pytest.mark.parametrize("instance,expected", [("7", 7), (" 7 ", 7), (7, 7), ("-3", -3), ("", None), (None, None)])
def test_instance_number_values(instance, expected):
    assert slice_key("p", FakeHeader(instance)).instance_number == expected
