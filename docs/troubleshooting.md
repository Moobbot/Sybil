# Troubleshooting Guide

This guide provides solutions for common issues you might encounter while using Sybil.

## Installation Issues

### Python Version

**Issue:** Python version mismatch
**Solution:**

```bash
# Check Python version
python --version  # Should be 3.10.x

# If incorrect version, install Python 3.10
# Windows: Download from python.org
# Linux:
sudo apt-get install python3.10
```

### Pip Version

**Issue:** Pip version too high
**Solution:**

```bash
# Downgrade pip to 24.0
pip install --upgrade pip==24.0
```

### Dependencies

**Issue:** Missing dependencies
**Solution:**

```bash
# Reinstall dependencies
python setup.py
```

## Model Issues

### Checkpoint Loading

**Issue:** Model checkpoints not found
**Solution:**

1. Check checkpoint directory:

```bash
ls sybil_checkpoints/
```

2. Download checkpoints:

```python
from call_model import download_checkpoints
download_checkpoints()
```

### GPU Issues

**Issue:** CUDA not available
**Solution:**

1. Check CUDA installation:

```bash
nvidia-smi
```

2. Install CUDA toolkit:

```bash
# Ubuntu
sudo apt-get install nvidia-cuda-toolkit
```

3. Use CPU version:

```bash
pip install -r requirements_cpu.txt
```

## API Issues

### Server Not Starting

**Issue:** Port already in use
**Solution:**

```bash
# Find process using port
netstat -ano | findstr :5555  # Windows
lsof -i :5555                 # Linux/Mac

# Kill process
taskkill /PID <pid> /F        # Windows
kill -9 <pid>                 # Linux/Mac
```

### File Upload Issues

**Issue:** File size too large
**Solution:**

1. Check file size limit in Flask:

```python
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
```

2. Compress files before upload

### Response Time

**Issue:** Slow predictions
**Solution:**

1. Enable GPU acceleration
2. Optimize batch size
3. Use caching

## File System Issues

### Permission Errors

**Issue:** Cannot write to directories
**Solution:**

```bash
# Set permissions
chmod -R 755 uploads/
chmod -R 755 results/
```

### Disk Space

**Issue:** Out of disk space
**Solution:**

1. Clean up old files:

```python
from utils import cleanup_old_results
cleanup_old_results(['uploads', 'results'])
```

2. Increase disk space
3. Implement file rotation

## Memory Issues

### Out of Memory

**Issue:** Memory errors during prediction
**Solution:**

1. Reduce batch size
2. Clear GPU memory:

```python
import torch
torch.cuda.empty_cache()
```

3. Monitor memory usage:

```python
import psutil
print(psutil.virtual_memory())
```

## Network Issues

### Connection Timeout

**Issue:** API requests timing out
**Solution:**

1. Check network connection
2. Increase timeout:

```python
requests.post(url, timeout=30)
```

3. Implement retry logic

### CORS Issues

**Issue:** Cross-origin requests blocked
**Solution:**

1. Enable CORS in Flask:

```python
from flask_cors import CORS
CORS(app)
```

2. Configure allowed origins

## Logging Issues

### Missing Logs

**Issue:** Logs not being written
**Solution:**

1. Check log configuration:

```python
LOG_CONFIG = {
    "FILE": "logs/sybil.log",
    "LEVEL": "DEBUG"
}
```

2. Create log directory:

```bash
mkdir -p logs
```

### Log Rotation

**Issue:** Log files too large
**Solution:**

1. Configure log rotation:

```python
LOG_CONFIG = {
    "MAX_BYTES": 10485760,  # 10MB
    "BACKUP_COUNT": 5
}
```

2. Implement manual rotation

## Docker Issues

### Container Not Starting

**Issue:** Docker container fails to start
**Solution:**

1. Check logs:

```bash
docker logs sybil
```

2. Verify environment variables
3. Check port mapping

### Volume Mounting

**Issue:** Cannot access mounted volumes
**Solution:**

1. Check volume permissions
2. Verify mount paths
3. Use absolute paths

## Performance Issues

### Slow Predictions

**Issue:** Model inference is slow
**Solution:**

1. Use GPU acceleration
2. Optimize model configuration
3. Implement caching

### High Memory Usage

**Issue:** Excessive memory consumption
**Solution:**

1. Monitor memory usage
2. Implement garbage collection
3. Optimize data loading

## Security Issues

### File Validation

**Issue:** Invalid file uploads
**Solution:**

1. Implement file validation:

```python
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
```

2. Add file type checking
3. Implement size limits

### API Security

**Issue:** Unauthorized access
**Solution:**

1. Implement authentication
2. Use HTTPS
3. Add rate limiting

## Support

If you encounter issues not covered in this guide:

1. Check the [GitHub Issues](https://github.com/reginabarzilaygroup/Sybil/issues)
2. Contact the maintainers
3. Submit a new issue with:
   - Error message
   - Steps to reproduce
   - System information
   - Log files
