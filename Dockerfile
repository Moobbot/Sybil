FROM python:3.10

WORKDIR /app

# Cài đặt các thư viện hệ thống cần thiết
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Tạo và sử dụng virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Cập nhật pip lên phiên bản 24.0
RUN pip install --upgrade pip==24.0

COPY requirements.txt setup.py .

# Cài đặt dependencies từ setup.py
RUN python setup.py

COPY . .


EXPOSE 5555

CMD ["python", "api.py"]