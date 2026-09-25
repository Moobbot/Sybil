"""P4c — tai checkpoint an toan khi bi ngat (can torch/sybil: chay trong image Sybil).

Trong container:  cd /app && python tests/test_download_checkpoints.py
"""
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


def _setup(tmp):
    import call_model as cm
    ckpt = os.path.join(tmp, "ckpt")
    os.makedirs(ckpt)
    names = ["a.ckpt", "b.ckpt", "cal.json"]
    src_zip = os.path.join(tmp, "src.zip")
    with zipfile.ZipFile(src_zip, "w") as z:
        for n in names:
            z.writestr(n, n * 1000)
    cm.FOLDERS["CHECKPOINT"] = ckpt
    cm.MODEL_PATHS[:] = [os.path.join(ckpt, n) for n in names[:2]]
    cm.urllib.request.urlretrieve = lambda url, dst: __import__("shutil").copy(src_zip, dst)
    return cm, ckpt, names


def test_tai_thanh_cong_file_nam_dung_cho_khong_con_thu_muc_tam():
    with tempfile.TemporaryDirectory() as tmp:
        cm, ckpt, names = _setup(tmp)
        cm.download_checkpoints()
        assert sorted(os.listdir(ckpt)) == sorted(names)
        assert open(os.path.join(ckpt, "a.ckpt")).read() == "a.ckpt" * 1000


def test_ngat_giua_luc_giai_nen_khong_de_lai_file_cut():
    with tempfile.TemporaryDirectory() as tmp:
        cm, ckpt, _ = _setup(tmp)
        real = zipfile.ZipFile.extractall

        def boom(self, path=None, *a, **k):
            # giai nen dung 1 file roi "mat dien"
            self.extract(self.namelist()[0], path)
            raise OSError("mat dien giua chung")
        zipfile.ZipFile.extractall = boom
        try:
            try:
                cm.download_checkpoints()
            except OSError:
                pass
        finally:
            zipfile.ZipFile.extractall = real
        assert os.listdir(ckpt) == [], os.listdir(ckpt)  # khong file nao o cho that, khong thu muc tam
        # lan sau: van thay "chua co" -> tai lai binh thuong
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
