FROM python:3.10

WORKDIR /app

COPY . .

RUN python setup.py

EXPOSE 5000

CMD ["python", "api.py"]