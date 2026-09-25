"""P4c — checkpoint download is safe when interrupted (needs torch/sybil: run inside the Sybil image).

Inside the container:  cd /app && python tests/test_download_checkpoints.py
"""
import contextlib
import os
import sys
import tempfile
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import pytest
    pytest.importorskip("torch")
except ImportError:
    pass


@contextlib.contextmanager
def _setup(tmp):
    """Point call_model at a temporary checkpoint folder and a fake download — and put
    everything back afterwards, so tests running later in the same session (the known-answer
    test) see the real checkpoints."""
    import call_model as cm
    saved = (cm.FOLDERS["CHECKPOINT"], list(cm.MODEL_PATHS), cm.urllib.request.urlretrieve)
    ckpt = os.path.join(tmp, "ckpt")
    os.makedirs(ckpt)
    names = ["a.ckpt", "b.ckpt", "cal.json"]
    src_zip = os.path.join(tmp, "src.zip")
    with zipfile.ZipFile(src_zip, "w") as z:
        for n in names:
            z.writestr(n, n * 1000)
    try:
        cm.FOLDERS["CHECKPOINT"] = ckpt
        cm.MODEL_PATHS[:] = [os.path.join(ckpt, n) for n in names[:2]]
        cm.urllib.request.urlretrieve = lambda url, dst: __import__("shutil").copy(src_zip, dst)
        yield cm, ckpt, names
    finally:
        cm.FOLDERS["CHECKPOINT"], cm.MODEL_PATHS[:], cm.urllib.request.urlretrieve = saved


def test_successful_download_puts_files_in_place_and_leaves_no_temp_dir():
    with tempfile.TemporaryDirectory() as tmp, _setup(tmp) as (cm, ckpt, names):
        cm.download_checkpoints()
        assert sorted(os.listdir(ckpt)) == sorted(names)
        assert open(os.path.join(ckpt, "a.ckpt")).read() == "a.ckpt" * 1000


def test_interrupted_extraction_leaves_no_truncated_file():
    with tempfile.TemporaryDirectory() as tmp, _setup(tmp) as (cm, ckpt, _):
        real = zipfile.ZipFile.extractall

        def boom(self, path=None, *a, **k):
            # extract exactly 1 file, then "lose power"
            self.extract(self.namelist()[0], path)
            raise OSError("power lost mid-way")
        zipfile.ZipFile.extractall = boom
        try:
            try:
                cm.download_checkpoints()
            except OSError:
                pass
        finally:
            zipfile.ZipFile.extractall = real
        assert os.listdir(ckpt) == [], os.listdir(ckpt)  # no file in the real location, no temp directory
        # next time: still seen as "missing" -> downloads again normally
        cm.download_checkpoints()
        assert all(os.path.exists(p) for p in cm.MODEL_PATHS)


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                fails += 1; print(f"FAIL {name}: {e!r}")
    sys.exit(1 if fails else 0)
