from flask import Flask

from config import HOST_CONNECT, PORT_CONNECT, RESULTS_FOLDER, UPLOAD_FOLDER
from routes import bp
# from utils import cleanup_old_results
from utils import get_local_ip

app = Flask(__name__)
app.register_blueprint(bp)

# cleanup_old_results([UPLOAD_FOLDER, RESULTS_FOLDER])

if __name__ == "__main__":
    LOCAL_IP = get_local_ip()
    print(f"Running on: http://127.0.0.1:{PORT_CONNECT} (localhost)")
    print(f"Running on: http://{LOCAL_IP}:{PORT_CONNECT} (local network)")

    # Run on all IP addresses (0.0.0.0) to accept both localhost and local network connections
    app.run(host=HOST_CONNECT, port=PORT_CONNECT, debug=True)
