import os

PYTHON_ENV = os.getenv("PYTHON_ENV", "develop")  # "production"

# Flask configuration
PORT_CONNECT = int(os.getenv("PORT_CONNECT", 5555))
HOST_CONNECT = "0.0.0.0"  # "0.0.0.0"

# Upload and results directories - support both relative and absolute paths
UPLOAD_FOLDER = os.getenv(
    "UPLOAD_FOLDER", os.path.join(os.path.dirname(__file__), "uploads")
)  # Upload directory
RESULTS_FOLDER = os.getenv(
    "RESULTS_FOLDER", os.path.join(os.path.dirname(__file__), "results")
)  # Results directory
CLEANUP_FOLDER = os.getenv(
    "CLEANUP_FOLDER", os.path.join(os.path.dirname(__file__), "cleanup")
)  # Results directory
CHECKPOINT_DIR = "sybil_checkpoints"
ALLOWED_EXTENSIONS = {"dcm", "png", "jpg", "jpeg"}

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(CLEANUP_FOLDER, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Checkpoint download URL
CHECKPOINT_URL = "https://dl.dropboxusercontent.com/scl/fi/56p7wa6ose6yxuzvj5ejh/sybil_checkpoints.zip?rlkey=13zvrawog90y6ntfst80sveg8&dl=1"

# List of model checkpoints
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

# Visualization configuration
VISUALIZATION_CONFIG = {
    # Attention ranking configuration
    "RANKING": {
        "DEFAULT_RETURN_TYPE": "top",  # 'all', 'top', or 'none'
        "DEFAULT_TOP_K": 6,  # Default number of top images
        # "MIN_SCORE": 0.0,  # Minimum attention score threshold
    },
}

# Model configuration
MODEL_CONFIG = {
    "RETURN_ATTENTIONS_DEFAULT": True,
    "WRITE_ATTENTION_IMAGES_DEFAULT": True,
    "SAVE_AS_DICOM_DEFAULT": True,
    "SAVE_ORIGINAL_DEFAULT": True,
}

# Prediction configuration - support both relative and absolute paths
PREDICTION_CONFIG = {
    "OVERLAY_PATH": "overlay",
    "PREDICTION_PATH": os.path.join(RESULTS_FOLDER, "prediction_scores.json"),
    "ATTENTION_PATH": os.path.join(RESULTS_FOLDER, "attention_scores.pkl"),
    "RANKING_PATH": os.path.join(RESULTS_FOLDER, "image_ranking.json"),
}
