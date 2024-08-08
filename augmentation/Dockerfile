FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /augmentation

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY . .
COPY fractals /data/fractals

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["/bin/bash", "-c"]
CMD ["tr -d '\\r' < /env_data/.env > /env_data/.env.unix && source /env_data/.env.unix && python augmentation.py"]