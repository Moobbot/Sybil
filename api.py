from flask import Flask
from routes import bp
from utils import cleanup_old_results, get_local_ip
from config import UPLOAD_FOLDER, RESULTS_FOLDER

app = Flask(__name__)
app.register_blueprint(bp)

cleanup_old_results([UPLOAD_FOLDER, RESULTS_FOLDER])

if __name__ == "__main__":
    LOCAL_IP = get_local_ip()
    HOST_CONNECT = (
        "0.0.0.0"  # Chạy trên tất cả các IP, bao gồm cả localhost và IP cục bộ
    )
    PORT_CONNECT = 5555
    print(f"Running on: http://127.0.0.1:{PORT_CONNECT} (localhost)")
    print(f"Running on: http://{LOCAL_IP}:{PORT_CONNECT} (local network)")

    # Chạy trên tất cả địa chỉ IP (0.0.0.0) để nhận cả localhost và IP cục bộ
    app.run(host=HOST_CONNECT, port=PORT_CONNECT, debug=True)
