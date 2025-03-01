FROM python:3.10

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /app

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY . .

RUN python setup.py

EXPOSE 5000

CMD ["python", "api.py"]