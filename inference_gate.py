"""Chạy suy luận MỘT ca một lúc; ca đến sau chờ tới lượt.

Dev server Flask chạy đa luồng: hai request cùng lúc là hai lần suy luận song
song, RAM đỉnh nhân đôi và máy bị OOM ("tràn máy"). Khoá này buộc các lần suy
luận chạy lần lượt. Request đến sau không bị từ chối, chỉ chờ.

`status()` báo trạng thái cho /health mà không phải chờ khoá, nên /health vẫn
trả lời ngay cả khi đang có ca chạy.

Watchdog: nếu một ca chạy quá INFERENCE_MAX_SECONDS thì coi là TREO (CUDA hang,
I/O đứng...). Khi đó khoá bị giữ mãi và mọi ca sau xếp hàng vô ích. /health báo
`stuck`, và process tự thoát để `restart: unless-stopped` trong compose đưa
service dậy lại — Docker KHÔNG tự khởi động lại container chỉ vì "unhealthy".
"""

import functools
import os
import sys
import threading
import time

# Ca hợp lệ dài nhất đo được: CVD trên CPU 748 s (P0). 30 phút là rộng rãi.
# Default có lý giải; chỉnh bằng biến môi trường nếu phần cứng khác.
INFERENCE_MAX_SECONDS = int(os.getenv("INFERENCE_MAX_SECONDS", "1800"))

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
        running = round(time.time() - started, 1) if started else 0
        return {
            "busy": _info["busy"],
            "waiting": _info["waiting"],
            "running_seconds": running,
            "stuck": running > INFERENCE_MAX_SECONDS,
        }


def _watchdog(check_every=30):
    while True:
        time.sleep(check_every)
        st = status()
        if st["stuck"]:
            print(
                f"[watchdog] suy luan chay {st['running_seconds']}s > "
                f"{INFERENCE_MAX_SECONDS}s — coi la TREO, thoat de Docker khoi dong lai",
                file=sys.stderr,
                flush=True,
            )
            os._exit(1)


def start_watchdog():
    """Gọi MỘT lần khi service khởi động."""
    threading.Thread(target=_watchdog, name="inference-watchdog", daemon=True).start()
