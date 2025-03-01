FROM python:3.10

WORKDIR /app

# Tạo và sử dụng virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Cập nhật pip lên phiên bản 24.0
RUN pip install --upgrade pip==24.0

COPY . .

# Cài đặt dependencies từ setup.py
RUN python setup.py

# Cài đặt các thư viện hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 5000

CMD ["python", "api.py"]