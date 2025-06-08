# API Documentation

This document provides detailed information about the Sybil API endpoints, request/response formats, and usage examples.

## Base URL

The API is accessible at:

- Development: `http://localhost:5555`
- Production: `http://<your-domain>:5555`

## Endpoints

### 1. Predict from Files

**Endpoint:** `/api_predict_file`  
**Method:** POST  
**Content-Type:** multipart/form-data

Upload individual files for prediction.

**Request:**

```bash
curl -X POST -F "file=@path/to/image1.dcm" -F "file=@path/to/image2.dcm" http://localhost:5555/api_predict_file
```

**Response:**

```json
{
    "session_id": "uuid-string",
    "predictions": [...],
    "overlay_images": [...],
    "attention_info": {...},
    "message": "Prediction successful."
}
```

### 2. Predict from ZIP

**Endpoint:** `/api_predict_zip`  
**Method:** POST  
**Content-Type:** multipart/form-data

Upload a ZIP file containing multiple images.

**Request:**

```bash
curl -X POST -F "file=@path/to/images.zip" http://localhost:5555/api_predict_zip
```

**Response:**

```json
{
    "session_id": "uuid-string",
    "predictions": [...],
    "overlay_images": "download-link",
    "attention_info": {...},
    "message": "Prediction successful."
}
```

### 3. Predict from Session

**Endpoint:** `/api_predict`  
**Method:** POST  
**Content-Type:** application/json

Make predictions using a previously created session.

**Request:**

```json
{
    "session_id": "uuid-string"
}
```

**Response:**

```json
{
    "session_id": "uuid-string",
    "predictions": [...],
    "attention_info": {...},
    "message": "Prediction successful."
}
```

### 4. Convert DICOM to PNG

**Endpoint:** `/convert-list`  
**Method:** POST  
**Content-Type:** multipart/form-data

Convert DICOM files to PNG format.

**Request:**

```bash
curl -X POST -F "files=@path/to/image1.dcm" -F "files=@path/to/image2.dcm" http://localhost:5555/convert-list
```

**Response:**

```json
{
    "images": [
        {
            "filename": "image1.dcm.png",
            "image_base64": "base64-encoded-image"
        }
    ]
}
```

### 5. Download Files

**Endpoint:** `/download/<session_id>/<filename>`  
**Method:** GET

Download a specific file from a session.

**Response:** File download

### 6. Download ZIP

**Endpoint:** `/download_zip/<session_id>`  
**Method:** GET

Download all files from a session as a ZIP archive.

**Response:** ZIP file download

### 7. Preview File

**Endpoint:** `/preview/<session_id>/<filename>`  
**Method:** GET

Preview a file in the browser.

**Response:** Image preview

### 8. Download GIF

**Endpoint:** `/download_gif/<session_id>`  
**Method:** GET

Download a GIF animation of the overlay images.

**Response:** GIF file download

## Error Responses

All endpoints return error responses in the following format:

```json
{
    "error": "Error message",
    "session_id": "uuid-string",  // if applicable
    "filename": "filename"        // if applicable
}
```

Common HTTP status codes:

- 200: Success
- 400: Bad Request
- 404: Not Found
- 500: Internal Server Error

## Rate Limiting

The API implements rate limiting to prevent abuse. Contact the administrators for specific limits.

## Authentication

Currently, the API does not require authentication. However, it's recommended to implement authentication for production use.

## Best Practices

1. Always check the response status code
2. Handle errors gracefully
3. Use appropriate content types
4. Clean up session data when no longer needed
5. Implement retry logic for failed requests
