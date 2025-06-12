import os
from pathlib import Path

# Environment Configuration
ENV_DEFAULT = "develop"
ENV = os.getenv("PYTHON_ENV", ENV_DEFAULT)
IS_DEV = ENV == "develop"

# Server Configuration
HOST_CONNECT_DEFAULT = "0.0.0.0"
PORT_CONNECT_DEFAULT = 5555
HOST_CONNECT = os.getenv("HOST_CONNECT", HOST_CONNECT_DEFAULT)
PORT_CONNECT = int(os.getenv("PORT_CONNECT", PORT_CONNECT_DEFAULT))

# Base Directory Configuration
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# Folder Configuration
FOLDERS = {
    "UPLOAD": os.getenv("UPLOAD_FOLDER", os.path.join(BASE_DIR, "uploads")),
    "RESULTS": os.getenv("RESULTS_FOLDER", os.path.join(BASE_DIR, "results")),
    "CLEANUP": os.getenv("CLEANUP_FOLDER", os.path.join(BASE_DIR, "cleanup")),
    "CHECKPOINT": os.path.join(BASE_DIR, "sybil_checkpoints"),
}

# Create necessary directories
for folder in FOLDERS.values():
    os.makedirs(folder, exist_ok=True)

# File Configuration
FILE_RETENTION = int(os.getenv("FILE_RETENTION", 3600))  # 1 hour
ALLOWED_EXTENSIONS = {"dcm", "png", "jpg", "jpeg"}

# Model Configuration
CHECKPOINT_URL = "https://github.com/Moobbot/Sybil/releases/download/1.8.3/sybil_checkpoints.zip"

MODEL_PATHS = [
    os.path.join(FOLDERS["CHECKPOINT"], f"{model}.ckpt")
    for model in [
        "28a7cd44f5bcd3e6cc760b65c7e0d54d",
        "56ce1a7d241dc342982f5466c4a9d7ef",
        "64a91b25f84141d32852e75a3aec7305",
        "65fd1f04cb4c5847d86a9ed8ba31ac1a",
        "624407ef8e3a2a009f9fa51f9846fe9a",
    ]
]

CALIBRATOR_PATH = os.path.join(
    FOLDERS["CHECKPOINT"], "sybil_ensemble_simple_calibrator.json"
)

MODEL_CONFIG = {
    "RETURN_ATTENTIONS_DEFAULT": True,
    "WRITE_ATTENTION_IMAGES_DEFAULT": True,
    "SAVE_AS_DICOM_DEFAULT": True,
    "SAVE_ORIGINAL_DEFAULT": True,
}

# Visualization Configuration
VISUALIZATION_CONFIG = {
    "RANKING": {
        "DEFAULT_RETURN_TYPE": "top",  # 'all', 'top', or 'none'
        "DEFAULT_TOP_K": 6,  # Default number of top images
    },
}

# Prediction Configuration
PREDICTION_CONFIG = {
    "OVERLAY_PATH": "overlay",
    "PREDICTION_PATH": os.path.join(FOLDERS["RESULTS"], "prediction_scores.json"),
    "ATTENTION_PATH": os.path.join(FOLDERS["RESULTS"], "attention_scores.pkl"),
    "RANKING_PATH": os.path.join(FOLDERS["RESULTS"], "image_ranking.json"),
}

# Logging Configuration
LOG_CONFIG = {
    "FILE": os.path.join(BASE_DIR, "logs", f"sybil_{ENV}.log"),
    "FORMAT": "%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
    "LEVEL": os.getenv("LOG_LEVEL", "DEBUG" if IS_DEV else "INFO"),
    "CONSOLE": IS_DEV,
    "MAX_BYTES": 10485760,  # 10MB
    "BACKUP_COUNT": 5,
}

# Error Messages
ERROR_MESSAGES = {
    "invalid_file": f"Invalid file format. Only {', '.join(ALLOWED_EXTENSIONS)} files are allowed.",
    "processing_error": "Error processing the files. Please check the file format and try again.",
    "server_error": "Internal server error. Please try again later.",
    "file_not_found": "Requested file not found.",
}

# Cleanup Configuration
CLEANUP_CONFIG = {
    "ENABLED": True,
    "INTERVAL_HOURS": 3,
    "MAX_AGE_DAYS": 1,
    "PATTERNS": {"UPLOAD": f"*.{ext}" for ext in ALLOWED_EXTENSIONS},
}

# Security Configuration
SECURITY_CONFIG = {
    "ALLOWED_IPS": ["127.0.0.1", "192.168.1.0/24", "10.0.0.0/8"],
    "CORS_ORIGINS": ["*"] if IS_DEV else ["https://example.com"],
    "CORS_METHODS": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    "CORS_HEADERS": ["*"],
}
