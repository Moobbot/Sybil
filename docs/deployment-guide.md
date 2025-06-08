# Deployment Guide

This guide provides instructions for deploying the Sybil application in various environments.

## Docker Deployment

### Prerequisites

- Docker installed
- Docker Compose (optional)
- NVIDIA Container Toolkit (for GPU support)

### Building the Docker Image

1. Build the image:

```bash
docker build -t sybil:latest .
```

2. Run the container:

```bash
docker run -d \
  --name sybil \
  -p 5555:5555 \
  -v /path/to/uploads:/app/uploads \
  -v /path/to/results:/app/results \
  sybil:latest
```

### Docker Compose Setup

Create a `docker-compose.yml` file:

```yaml
version: '3'
services:
  sybil:
    build: .
    ports:
      - "5555:5555"
    volumes:
      - ./uploads:/app/uploads
      - ./results:/app/results
    environment:
      - HOST_CONNECT=0.0.0.0
      - PORT_CONNECT=5555
      - PYTHON_ENV=prod
    restart: unless-stopped
```

Run with:

```bash
docker-compose up -d
```

## Production Deployment

### Environment Setup

1. Set environment variables:

```bash
export HOST_CONNECT=0.0.0.0
export PORT_CONNECT=5555
export PYTHON_ENV=prod
export UPLOAD_FOLDER=/path/to/uploads
export RESULTS_FOLDER=/path/to/results
```

2. Create necessary directories:

```bash
mkdir -p /path/to/uploads
mkdir -p /path/to/results
```

### System Requirements

- Python 3.10
- CUDA-compatible GPU (recommended)
- 16GB RAM minimum
- 50GB free disk space

### Security Considerations

1. **File Permissions**

```bash
chmod 755 /path/to/uploads
chmod 755 /path/to/results
```

2. **Firewall Configuration**

```bash
# Allow incoming traffic on port 5555
sudo ufw allow 5555/tcp
```

3. **SSL/TLS Setup**

- Use a reverse proxy (e.g., Nginx)
- Configure SSL certificates
- Enable HTTPS

### Nginx Configuration

Example Nginx configuration:

```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:5555;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Monitoring

### Logging

1. Configure logging in `config.py`:

```python
LOG_CONFIG = {
    "FILE": "/path/to/logs/sybil.log",
    "FORMAT": "%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
    "LEVEL": "INFO",
    "MAX_BYTES": 10485760,  # 10MB
    "BACKUP_COUNT": 5
}
```

2. Set up log rotation:

```bash
sudo logrotate -f /etc/logrotate.d/sybil
```

### Health Checks

1. Create a health check endpoint:

```python
@app.route("/health")
def health_check():
    return jsonify({"status": "healthy"})
```

2. Monitor with:

```bash
curl http://localhost:5555/health
```

## Backup and Recovery

### Backup Strategy

1. Regular backups of:
   - Model checkpoints
   - Configuration files
   - Upload and result directories

2. Automated backup script:

```bash
#!/bin/bash
backup_dir="/path/to/backups"
timestamp=$(date +%Y%m%d_%H%M%S)
tar -czf "$backup_dir/sybil_$timestamp.tar.gz" \
    /path/to/uploads \
    /path/to/results \
    /path/to/sybil_checkpoints
```

### Recovery Procedure

1. Stop the service:

```bash
docker-compose down  # or systemctl stop sybil
```

2. Restore from backup:

```bash
tar -xzf backup_file.tar.gz -C /path/to/restore
```

3. Restart the service:

```bash
docker-compose up -d  # or systemctl start sybil
```

## Scaling

### Horizontal Scaling

1. Use a load balancer
2. Deploy multiple instances
3. Share storage between instances

### Vertical Scaling

1. Increase server resources
2. Optimize model performance
3. Use GPU acceleration

## Troubleshooting

Common issues and solutions:

1. **Memory Issues**
   - Increase swap space
   - Optimize batch size
   - Monitor memory usage

2. **Performance Issues**
   - Enable GPU support
   - Optimize model configuration
   - Use caching

3. **Storage Issues**
   - Implement cleanup routines
   - Monitor disk usage
   - Configure proper permissions
