import os

PYTHON_ENV = "develop" # "production"
# Cấu hình Flask
PORT_CONNECT = 5555 # 5555
HOST_CONNECT = "0.0.0.0" # "0.0.0.0"
UPLOAD_FOLDER = "uploads"
RESULTS_FOLDER = "results"
CHECKPOINT_DIR = "sybil_checkpoints"
ALLOWED_EXTENSIONS = {"dcm", "png", "jpg", "jpeg"}

# Tạo thư mục nếu chưa có
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# URL tải checkpoint
CHECKPOINT_URL = "https://github.com/reginabarzilaygroup/Sybil/releases/download/v1.5.0/sybil_checkpoints.zip"

# Danh sách checkpoint mô hình
MODEL_PATHS = [
    os.path.join(CHECKPOINT_DIR, f"{model}.ckpt")
    for model in [
        "28a7cd44f5bcd3e6cc760b65c7e0d54d",
        "56ce1a7d241dc342982f5466c4a9d7ef",
        "64a91b25f84141d32852e75a3aec7305",
        "65fd1f04cb4c5847d86a9ed8ba31ac1a",
        "624407ef8e3a2a009f9fa51f9846fe9a",
    ]
]

CALIBRATOR_PATH = os.path.join(CHECKPOINT_DIR, "sybil_ensemble_simple_calibrator.json")

# Cấu hình Visualization
VISUALIZATION_CONFIG = {
    # Cấu hình attention ranking
    "RANKING": {
        "DEFAULT_RETURN_TYPE": "top",  # 'all', 'top', hoặc 'none'
        "DEFAULT_TOP_K": 6,  # Số lượng ảnh top mặc định
        # "MIN_SCORE": 0.0,  # Điểm attention tối thiểu để xem xét
    },
}

# Cấu hình Model
MODEL_CONFIG = {
    "RETURN_ATTENTIONS_DEFAULT": True,
    "WRITE_ATTENTION_IMAGES_DEFAULT": True,
    "SAVE_AS_DICOM_DEFAULT": True,
    "SAVE_ORIGINAL_DEFAULT": True,
}

PREDICTION_CONFIG = {
    "OVERLAY_PATH": "overlay", # "serie_0"
    "PREDICTION_PATH": os.path.join(RESULTS_FOLDER, "prediction_scores.json"),
    "ATTENTION_PATH": os.path.join(RESULTS_FOLDER, "attention_scores.pkl"),
    "RANKING_PATH": os.path.join(RESULTS_FOLDER, "image_ranking.json"),
}
