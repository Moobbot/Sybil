"""Chạy suy luận MỘT ca một lúc; ca đến sau chờ tới lượt.

Dev server Flask chạy đa luồng: hai request cùng lúc là hai lần suy luận song
song, RAM đỉnh nhân đôi và máy bị OOM ("tràn máy"). Khoá này buộc các lần suy
luận chạy lần lượt. Request đến sau không bị từ chối, chỉ chờ.

`status()` báo trạng thái cho /health mà không phải chờ khoá, nên /health vẫn
trả lời ngay cả khi đang có ca chạy.
"""

import functools
import threading
import time

_run_lock = threading.Lock()
_info_lock = threading.Lock()
_info = {"busy": False, "started_at": None, "waiting": 0}


def serialized(fn):
    """Bọc một hàm suy luận để các lần gọi chạy nối tiếp nhau."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with _info_lock:
            _info["waiting"] += 1
        _run_lock.acquire()
        try:
            with _info_lock:
                _info["waiting"] -= 1
                _info["busy"] = True
                _info["started_at"] = time.time()
            return fn(*args, **kwargs)
        finally:
            with _info_lock:
                _info["busy"] = False
                _info["started_at"] = None
            _run_lock.release()

    return wrapper


def status():
    """Trạng thái hiện tại; không chờ khoá suy luận."""
    with _info_lock:
        started = _info["started_at"]
        return {
            "busy": _info["busy"],
            "waiting": _info["waiting"],
            "running_seconds": round(time.time() - started, 1) if started else 0,
        }
