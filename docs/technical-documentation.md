# Technical Documentation

## Overview

This document provides comprehensive technical documentation for the custom Sybil implementation, including system architecture, API endpoints, configuration, and deployment details.

## System Architecture

### Core Components

1. **API Server (`api.py`)**
   - Flask-based REST API server
   - Handles HTTP requests and responses
   - Manages file uploads and predictions

2. **Model Handler (`call_model.py`)**
   - Manages model loading and predictions
   - Handles DICOM and PNG file processing
   - Processes attention scores and visualizations

3. **Route Handler (`routes.py`)**
   - Defines API endpoints
   - Manages file uploads and downloads
   - Handles session management

4. **Utility Functions (`utils.py`)**
   - File handling and conversion
   - DICOM to PNG conversion
   - File cleanup and management

### Configuration (`config.py`)

```python
# Key Configuration Parameters
ENV = "develop" | "production"
HOST_CONNECT = "0.0.0.0"
PORT_CONNECT = 5555
FILE_RETENTION = 3600  # 1 hour
ALLOWED_EXTENSIONS = {"dcm", "png", "jpg", "jpeg"}
```

## API Endpoints

### 1. Prediction Endpoints

#### POST `/api_predict`

- **Purpose**: Process pre-extracted files
- **Input**: `session_id`
- **Output**: Prediction results with attention information

#### POST `/api_predict_file`

- **Purpose**: Process uploaded files
- **Input**: Multiple files
- **Output**: Prediction results with overlay images

#### POST `/api_predict_zip`

- **Purpose**: Process ZIP archive
- **Input**: ZIP file containing DICOM/PNG files
- **Output**: Prediction results with ZIP download link

### 2. File Management Endpoints

#### GET `/download/<session_id>/<filename>`

- **Purpose**: Download overlay image
- **Input**: Session ID and filename
- **Output**: File download

#### GET `/download_zip/<session_id>`

- **Purpose**: Download ZIP archive
- **Input**: Session ID
- **Output**: ZIP file download

#### GET `/preview/<session_id>/<filename>`

- **Purpose**: Preview overlay image
- **Input**: Session ID and filename
- **Output**: Image preview

#### GET `/download_gif/<session_id>`

- **Purpose**: Download GIF animation
- **Input**: Session ID
- **Output**: GIF file download

## Installation

### Prerequisites

1. Python 3.10
2. pip version <= 24.0
3. CUDA-compatible GPU (recommended)

### Installation Steps

1. Clone repository:

```bash
git clone https://github.com/Moobbot/Sybil.git
cd Sybil
```

2. Install dependencies:

```bash
pip install --upgrade pip==24.0
python setup.py
```

3. Configure environment:

```bash
export CUSTOM_ENV=development
export CUSTOM_LOG_LEVEL=DEBUG
```

## Docker Deployment

### Build and Run

```bash
# Build Docker image
docker build -t sybil-custom:v1 .

# Run container
docker run -d -p 5555:5555 sybil-custom:v1
```

### Environment Variables

```dockerfile
ENV HOST_CONNECT=0.0.0.0 \
    PORT=5555 \
    ENV=prod \
    DEVICE=cuda
```

## Model Configuration

### Model Settings

```python
MODEL_CONFIG = {
    "RETURN_ATTENTIONS_DEFAULT": True,
    "WRITE_ATTENTION_IMAGES_DEFAULT": True,
    "SAVE_AS_DICOM_DEFAULT": True,
    "SAVE_ORIGINAL_DEFAULT": True
}
```

### Visualization Settings

```python
VISUALIZATION_CONFIG = {
    "RANKING": {
        "DEFAULT_RETURN_TYPE": "top",
        "DEFAULT_TOP_K": 6
    }
}
```

## File Processing

### Supported Formats

- DICOM (.dcm)
- PNG (.png)
- JPEG (.jpg, .jpeg)

### File Retention

- Default retention period: 1 hour
- Configurable via `FILE_RETENTION` environment variable

## Security

### Access Control

```python
SECURITY_CONFIG = {
    "ALLOWED_IPS": ["127.0.0.1", "192.168.1.0/24", "10.0.0.0/8"],
    "CORS_ORIGINS": ["*"] if IS_DEV else ["https://example.com"],
    "CORS_METHODS": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    "CORS_HEADERS": ["*"]
}
```

## Error Handling

### Common Error Messages

```python
ERROR_MESSAGES = {
    "invalid_file": "Invalid file format. Only dcm, png, jpg, jpeg files are allowed.",
    "processing_error": "Error processing the files. Please check the file format and try again.",
    "server_error": "Internal server error. Please try again later.",
    "file_not_found": "Requested file not found."
}
```

## Logging

### Log Configuration

```python
LOG_CONFIG = {
    "FILE": "logs/sybil_{ENV}.log",
    "FORMAT": "%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
    "LEVEL": "DEBUG" if IS_DEV else "INFO",
    "CONSOLE": IS_DEV,
    "MAX_BYTES": 10485760,  # 10MB
    "BACKUP_COUNT": 5
}
```

## Dependencies

### Core Requirements

```
Flask==3.1.0
numpy==1.24.1
pydicom==2.3.0
pillow==11.1.0
torch==1.13.1
torchvision==0.14.1
```

### GPU Requirements

- CUDA-compatible GPU
- CUDA Toolkit 11.8 or 12.1
- cuDNN compatible with CUDA version

## Support

For technical support:

- Email: <ngotam24082001@gmail.com>
- GitHub: [Moobbot/Sybil](https://github.com/Moobbot/Sybil)
- Documentation: [Custom Documentation](./custom-implementation.md)
