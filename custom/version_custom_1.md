# Changelog

- Added new imports for various libraries including `base64`, `io`, `json`, `os`, `pickle`, `shutil`, `socket`, `time`, `typing`, `uuid`, `urllib`, `zipfile`, `numpy`, `pydicom`, `PIL`, `Flask`, and `werkzeug`.
- Configured Flask application with upload and results directories.
- Implemented functions to handle file uploads, download checkpoints, clean up old results, save uploaded files, load models, and predict using the Sybil model.
- Added API endpoints for prediction, file download, file preview, GIF download, and DICOM to PNG conversion.
- Improved logging and error handling throughout the application.
- Enhanced prediction function to support DICOM and PNG file types, and added options for saving predictions as DICOM images.
- Introduced session management using UUIDs for each prediction request.
- Added functionality to visualize attention maps and save them as overlay images.
- Implemented local IP address retrieval for better network configuration.
- Updated application to run on all IP addresses, including localhost and local network.
