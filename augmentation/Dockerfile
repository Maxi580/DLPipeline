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

RUN chmod +x load_env.sh
ENTRYPOINT ["./load_env.sh"]
CMD ["python", "augmentation.py"]