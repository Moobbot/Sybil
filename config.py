import os

# Cấu hình Flask
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