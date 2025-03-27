# Constants for visualization
VISUALIZATION_CONFIG = {
    # General constants
    "EPS": 1e-6,  # Epsilon value for numerical stability
    "DEFAULT_GAIN": 3,  # Hệ số khuếch đại mặc định cho attention map
    "DEFAULT_ATTENTION_THRESHOLD": 1e-6,  # Ngưỡng attention mặc định
    "IMAGE_SIZE": (512, 512),  # Kích thước ảnh chuẩn
    # Cấu hình DICOM
    "DICOM": {
        "BITS_ALLOCATED": 8,
        "BITS_STORED": 8,
        "HIGH_BIT": 7,
        "SAMPLES_PER_PIXEL": 3,
        "PHOTOMETRIC_INTERPRETATION": "RGB",
        "PLANAR_CONFIGURATION": 0,
        "PIXEL_REPRESENTATION": 0,
        "RESCALE_INTERCEPT": 0,
        "RESCALE_SLOPE": 1,
        "VOI_LUT_FUNCTION": "LINEAR",
        # Các thuộc tính DICOM cần sao chép
        "ATTRIBUTES_TO_COPY": [
            "PixelSpacing",
            "SliceLocation",
            "ImagePositionPatient",
            "ImageOrientationPatient",
            "InstanceNumber",
        ],
    },
    # Cấu hình tên file
    "FILE_NAMING": {
        "PREDICTION_PREFIX": "pred_",
        "ORIGINAL_SUFFIX": "", #_original
        "DEFAULT_PATIENT": "Unknown_Patient",
    },
    # Cấu hình logging
    "LOGGING": {
        "SAVE_ORIGINAL_MESSAGE": "Saving original image for slice {} (no significant attention)",
        "MISSING_METADATA_WARNING": "No corresponding DICOM metadata found",
        "MISSING_ATTENTION_WARNING": "Missing attention data for series {}",
    },
}
