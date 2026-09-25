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
# config.py doc PYTHON_ENV, KHONG doc ENV. Truoc day dat ENV=prod (sai ten) nen
# IS_DEV=True -> debug + reloader -> model nap 2 lan (P0 do: 2 tien trinh, RAM dinh 6.5 GiB).
ENV HOST_CONNECT=0.0.0.0 \
    PORT=5555 \
    PYTHON_ENV=prod \
    DEVICE=cuda

EXPOSE 5555

CMD ["python", "api.py"]