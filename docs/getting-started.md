# Getting Started with Sybil

This guide will help you get up and running with Sybil, a lung cancer risk prediction system.

## Prerequisites

- Python 3.10
- pip version <= 24.0
- CUDA-compatible GPU (optional, for faster processing)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Moobbot/Sybil.git
cd Sybil
```

2. Upgrade pip to version 24.0:

```bash
pip install --upgrade pip==24.0
```

3. Install dependencies:

```bash
python setup.py
```

## Basic Usage

### Starting the API Server

1. Navigate to the project directory:

```bash
cd Sybil
```

2. Start the API server:

```bash
python api.py
```

The server will start on:

- Localhost: <http://127.0.0.1:5555>
- Local network: http://<your-ip>:5555

### Making Predictions

You can make predictions using the API in several ways:

1. **Direct File Upload**:

```bash
curl -X POST -F "file=@path/to/your/image.dcm" http://localhost:5555/api_predict_file
```

2. **ZIP File Upload**:

```bash
curl -X POST -F "file=@path/to/your/images.zip" http://localhost:5555/api_predict_zip
```

3. **Using Session ID**:

```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"session_id": "your-session-id"}' \
     http://localhost:5555/api_predict
```

## Supported File Types

- DICOM (.dcm)
- PNG (.png)
- JPEG (.jpg, .jpeg)

## Environment Variables

You can configure the application using the following environment variables:

- `HOST_CONNECT`: Server host (default: "0.0.0.0")
- `PORT_CONNECT`: Server port (default: 5555)
- `PYTHON_ENV`: Environment type (default: "develop")
- `UPLOAD_FOLDER`: Upload directory path
- `RESULTS_FOLDER`: Results directory path
- `FILE_RETENTION`: File retention time in seconds (default: 3600)

## Next Steps

- Read the [API Documentation](./api-documentation.md) for detailed endpoint information
- Check the [Model Documentation](./model-documentation.md) to understand the prediction system
- Review the [Deployment Guide](./deployment-guide.md) for production setup
