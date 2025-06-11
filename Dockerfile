FROM python:3.10

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip==24.0

# Copy only requirements first to leverage Docker cache
COPY requirements.txt setup.py ./

# Install dependencies
RUN python setup.py

# Copy the rest of the application
COPY . .

# Set environment variables
ENV HOST_CONNECT=0.0.0.0 \
    PORT=5555 \
    ENV=prod \
    DEVICE=cuda

EXPOSE 5555

CMD ["python", "api.py"]