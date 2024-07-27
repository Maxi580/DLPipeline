FROM python:3.12-slim

WORKDIR /augmentation

COPY augmentation/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY augmentation .
COPY augmentation/fractals /data/fractals

ENV PYTHONUNBUFFERED=1

CMD ["python", "augmentation.py"]