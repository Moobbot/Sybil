FROM python:3.10

WORKDIR /app

# Cài đặt các thư viện hệ thống cần thiết
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Cập nhật pip lên phiên bản 24.0
RUN pip install --upgrade pip==24.0

COPY requirements.txt setup.py .

# Cài đặt dependencies từ setup.py
RUN python setup.py

COPY . .

# Thiết lập biến môi trường cho ứng dụng
ENV HOST_CONNECT=0.0.0.0 \
    PORT=5555 \
    ENV=prod \
    DEVICE=cuda

EXPOSE 5555

CMD ["python", "api.py"]