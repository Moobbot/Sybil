# Constants for visualization
VISUALIZATION_CONFIG = {
    # General constants
    'EPS': 1e-6,  # Epsilon value for numerical stability
    
    # Image processing
    'DEFAULT_GAIN': 3,
    'DEFAULT_ATTENTION_THRESHOLD': 1e-6,
    'DEFAULT_IMAGE_SIZE': (512, 512),
    
    # DICOM specific
    'DICOM_BITS_ALLOCATED': 8,
    'DICOM_BITS_STORED': 8,
    'DICOM_HIGH_BIT': 7,
    'DICOM_SAMPLES_PER_PIXEL': 3,
    'DICOM_PHOTOMETRIC_INTERPRETATION': 'RGB',
    'DICOM_PLANAR_CONFIGURATION': 0,
    'DICOM_PIXEL_REPRESENTATION': 0,
    'DICOM_RESCALE_INTERCEPT': 0,
    'DICOM_RESCALE_SLOPE': 1,
    'DICOM_VOI_LUT_FUNCTION': 'LINEAR',
    
    # Important DICOM attributes to copy
    'DICOM_ATTRIBUTES_TO_COPY': [
        'PixelSpacing',
        'SliceLocation',
        'ImagePositionPatient',
        'ImageOrientationPatient',
        'InstanceNumber',
    ]
} 